"""
Parameter-GRU: conditional GRU regressor for curve parameters (k, s, A).

Architecture
------------
Decoder-only GRU conditioned on (svara_type, total_dur).
At each step receives: seg_type_oh | dur_rel | delta_norm | slope_prev_norm
Output: (log_k_raw, logit_s_raw, A_raw) per step, masked to STA/TR only.

Diversity at generation time is injected via per-svara residual Gaussians
fitted post-training (see fit_residuals / save_residuals / load_residuals).

B2 loss enforces derivative continuity at consecutive STA/TR boundaries.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.models.param_gru.dataset_param_gru import (
    DEC_INPUT_DIM, OUTPUT_DIM,
    K_MIN, K_MAX, S_MIN, S_MAX,
    A_SCALE, LOG_K_SCALE, LOGIT_S_SCALE, SLOPE_SCALE, M_SCALE,
)
from src.models.gruvae.model_gruvae import lengths_to_mask  # noqa: F401

_N_SVARA = 7

# Type index → name for STA/TR segments that have curve params
_IDX_TO_SEG = {2: "STAp", 3: "STAt", 4: "TRa", 5: "TRd"}

# imported lazily inside generate() to avoid circular import at module level
def _svara_labels() -> list[str]:
    from src.models.param_gru.dataset_param_gru import SVARA_LABELS  # type: ignore
    return SVARA_LABELS


# ── config ────────────────────────────────────────────────────────────────────

@dataclass
class ParamGRUConfig:
    dec_input_dim:  int   = DEC_INPUT_DIM    # 9
    output_dim:     int   = OUTPUT_DIM       # 3
    hidden_dim:     int   = 64
    num_layers:     int   = 1
    cond_dim:       int   = 8               # svara_oh(7) + log1p(total_dur)(1)
    lambda_slope:   float = 0.05            # B2 loss weight


# ── residual distributions ────────────────────────────────────────────────────

@dataclass
class ResidualDist:
    """
    Trivariate Gaussian over [log_k_raw, logit_s_raw, A_raw] prediction residuals.
    mean ≈ 0 for a well-calibrated model; cov captures per-svara variability.
    """
    mean: np.ndarray   # (3,)
    cov:  np.ndarray   # (3, 3)

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return rng.multivariate_normal(self.mean, self.cov).astype(np.float32)


def fit_residuals(
    model: "ParamGRU",
    loader,
    device: torch.device,
    min_samples: int = 10,
) -> dict[str, ResidualDist]:
    """
    Run model over loader and fit a per-svara Gaussian to prediction residuals
    in raw output space [log_k_raw, logit_s_raw, A_raw].

    Returns dict keyed by svara label + "pooled" fallback.
    """
    svara_labels = _svara_labels()
    model.eval()
    # Keys: "{svara}_{seg_type}" (e.g. "S_STAp") and per-type pooled "pooled_{seg_type}"
    buckets: dict[str, list] = defaultdict(list)

    with torch.no_grad():
        for batch in loader:
            batch_d = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                       for k, v in batch.items()}
            pred    = model(batch_d)["params"].cpu().numpy()       # (B, T, 3)
            gt      = batch_d["targets"].cpu().numpy()             # (B, T, 3)
            mask    = batch_d["target_mask"].cpu().numpy()         # (B, T)
            sidx    = batch_d["svara_idx"].cpu().numpy()           # (B,)
            # seg_type_oh is the first 6 dims of dec_input
            seg_idx = batch_d["dec_input"][:, :, :6].argmax(dim=-1).cpu().numpy()  # (B, T)

            for b in range(pred.shape[0]):
                sv = svara_labels[int(sidx[b])]
                for t in range(pred.shape[1]):
                    if not mask[b, t]:
                        continue
                    seg_name = _IDX_TO_SEG.get(int(seg_idx[b, t]), "unknown")
                    resid    = pred[b, t] - gt[b, t]  # (3,)
                    buckets[f"{sv}_{seg_name}"].append(resid)
                    buckets[f"pooled_{seg_name}"].append(resid)
                    buckets["pooled"].append(resid)

    dists: dict[str, ResidualDist] = {}

    for key, chunks in buckets.items():
        R = np.array(chunks)              # (n, 3)
        if len(R) < min_samples:
            continue
        dists[key] = ResidualDist(
            mean=R.mean(axis=0),
            cov=(np.cov(R.T) if len(R) >= 2 else np.eye(3)) + np.eye(3) * 1e-6,
        )

    return dists


def save_residuals(dists: dict[str, ResidualDist], path: Path | str) -> None:
    data = {}
    for key, d in dists.items():
        data[f"{key}_mean"] = d.mean
        data[f"{key}_cov"]  = d.cov
    np.savez(path, **data)


def load_residuals(path: Path | str) -> dict[str, ResidualDist]:
    data = np.load(path)
    keys = {k[:-5] for k in data.files if k.endswith("_mean")}
    return {k: ResidualDist(mean=data[f"{k}_mean"], cov=data[f"{k}_cov"]) for k in keys}


# ── differentiable slopes ─────────────────────────────────────────────────────

def _norm_slope_torch(
    log_k_raw:   torch.Tensor,
    logit_s_raw: torch.Tensor,
    A_raw:       torch.Tensor,
    at_end:      bool,
) -> torch.Tensor:
    """Normalized derivative of curve_model at t=0 or t=1 (dimensionless).

    Multiply by delta_cents / dur_sec to obtain the physical slope in ¢/s.
    """
    k = torch.exp(log_k_raw * LOG_K_SCALE).clamp(K_MIN, K_MAX)
    s = torch.sigmoid(logit_s_raw * LOGIT_S_SCALE).clamp(S_MIN, S_MAX)
    A = A_raw * A_SCALE

    h0    = torch.tanh(-k * s)
    h1    = torch.tanh(k * (1.0 - s))
    denom = (h1 - h0).clamp(min=1e-9)

    arg = k * (1.0 - s) if at_end else k * s
    dh  = k / torch.cosh(arg).pow(2) - 2.0 * math.pi * A
    return dh / denom


def _slope_torch(
    log_k_raw:   torch.Tensor,
    logit_s_raw: torch.Tensor,
    A_raw:       torch.Tensor,
    delta_cents: torch.Tensor,
    dur_sec:     torch.Tensor,
    at_end:      bool,
) -> torch.Tensor:
    k = torch.exp(log_k_raw * LOG_K_SCALE).clamp(K_MIN, K_MAX)
    s = torch.sigmoid(logit_s_raw * LOGIT_S_SCALE).clamp(S_MIN, S_MAX)
    A = A_raw * A_SCALE

    h0    = torch.tanh(-k * s)
    h1    = torch.tanh(k * (1.0 - s))
    denom = (h1 - h0).clamp(min=1e-9)

    arg = k * (1.0 - s) if at_end else k * s
    dh  = k / torch.cosh(arg).pow(2) - 2.0 * math.pi * A
    return (dh / denom) * delta_cents / dur_sec.clamp(min=1e-4)


# ── decoder ───────────────────────────────────────────────────────────────────

class ParamDecoder(nn.Module):
    def __init__(self, cfg: ParamGRUConfig):
        super().__init__()
        self.cfg = cfg
        gru_in   = cfg.dec_input_dim + cfg.cond_dim
        self.gru = nn.GRU(gru_in, cfg.hidden_dim, num_layers=cfg.num_layers, batch_first=True)
        self.init_hidden = nn.Sequential(
            nn.Linear(cfg.cond_dim, cfg.hidden_dim * cfg.num_layers),
            nn.Tanh(),
        )
        self.param_head = nn.Linear(cfg.hidden_dim, cfg.output_dim)

    def _h0(self, cond: torch.Tensor) -> torch.Tensor:
        B = cond.size(0)
        h = self.init_hidden(cond)
        return h.view(B, self.cfg.num_layers, self.cfg.hidden_dim).transpose(0, 1).contiguous()

    def forward(
        self, cond: torch.Tensor, dec_input: torch.Tensor, lengths: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _ = dec_input.shape
        cond_exp = cond.unsqueeze(1).expand(B, T, -1)
        gru_in   = torch.cat([dec_input, cond_exp], dim=-1)

        h0     = self._h0(cond)
        packed = pack_padded_sequence(gru_in, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed, h0)
        out, _ = pad_packed_sequence(out, batch_first=True)

        if out.size(1) < T:
            pad = torch.zeros(B, T - out.size(1), self.cfg.hidden_dim, device=out.device)
            out = torch.cat([out, pad], dim=1)

        return self.param_head(out)  # (B, T, 3)

    def generate_step(
        self, x_t: torch.Tensor, hidden: torch.Tensor, cond: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gru_in = torch.cat([x_t, cond], dim=-1).unsqueeze(1)
        out, hidden = self.gru(gru_in, hidden)
        return self.param_head(out.squeeze(1)), hidden


# ── full model ────────────────────────────────────────────────────────────────

class ParamGRU(nn.Module):
    def __init__(self, cfg: ParamGRUConfig):
        super().__init__()
        self.cfg     = cfg
        self.decoder = ParamDecoder(cfg)

    def _make_cond(self, svara_idx: torch.Tensor, total_dur: torch.Tensor) -> torch.Tensor:
        oh    = F.one_hot(svara_idx, num_classes=_N_SVARA).float()
        dur_f = torch.log1p(total_dur.float()).unsqueeze(-1)
        return torch.cat([oh, dur_f], dim=-1)  # (B, 8)

    def forward(self, batch: dict) -> dict:
        cond   = self._make_cond(batch["svara_idx"], batch["total_dur"])
        params = self.decoder(cond, batch["dec_input"], batch["lengths"])
        return {"params": params}

    def generate(
        self,
        seg_type_oh:        torch.Tensor,                  # (B, T, 6)
        dur_rel:            torch.Tensor,                  # (B, T)
        delta_norm:         torch.Tensor,                  # (B, T)
        dur_sec:            torch.Tensor,                  # (B, T)
        delta_cents:        torch.Tensor,                  # (B, T)
        sta_tr_mask:        torch.Tensor,                  # (B, T) bool
        lengths:            torch.Tensor,                  # (B,)
        svara_idx:          torch.Tensor,                  # (B,)
        total_dur:          torch.Tensor,                  # (B,)
        residual_dists:     dict[str, ResidualDist] | None = None,
        rng:                np.random.Generator | None = None,
        dy0_required_init:  torch.Tensor | None = None,   # (B, T) boundary init slopes
    ) -> dict:
        """
        Autoregressive generation.

        If residual_dists is provided, adds per-svara noise sampled from the
        fitted residual Gaussian at each STA/TR step, giving diversity
        calibrated on the empirical spread of the training data.
        """
        svara_labels = _svara_labels()
        if rng is None and residual_dists is not None:
            rng = np.random.default_rng()

        B, T, _ = seg_type_oh.shape
        device  = seg_type_oh.device

        cond   = self._make_cond(svara_idx.to(device), total_dur.to(device))
        hidden = self.decoder._h0(cond)

        # Precompute svara label per batch element for residual lookup
        sv_labels = [svara_labels[int(svara_idx[b].item())] for b in range(B)]

        dy0_required = torch.zeros(B, device=device)
        all_params   = []
        all_slopes   = []

        for t in range(T):
            # Override dy0_required with pre-computed init when nonzero.
            # Using "nonzero" rather than "dy0_required==0" so actual CPVAE
            # boundary slopes always take precedence over the chain value.
            if dy0_required_init is not None:
                init_val = dy0_required_init[:, t]
                use_init = (init_val != 0.0) & sta_tr_mask[:, t]
                dy0_required = torch.where(use_init, init_val, dy0_required)

            dy0_req_n = torch.tanh(dy0_required / M_SCALE).unsqueeze(-1)
            x_t = torch.cat([
                seg_type_oh[:, t, :],
                dur_rel[:, t].unsqueeze(-1),
                delta_norm[:, t].unsqueeze(-1),
                dy0_req_n,
            ], dim=-1)

            params_t, hidden = self.decoder.generate_step(x_t, hidden, cond)

            if residual_dists is not None:
                seg_type_t = seg_type_oh[:, t, :].argmax(dim=-1).cpu().numpy()  # (B,)
                for b in range(B):
                    if sta_tr_mask[b, t]:
                        sv       = sv_labels[b]
                        seg_name = _IDX_TO_SEG.get(int(seg_type_t[b]), "unknown")
                        dist = (
                            residual_dists.get(f"{sv}_{seg_name}")
                            or residual_dists.get(f"pooled_{seg_name}")
                            or residual_dists.get(sv)
                            or residual_dists.get("pooled")
                        )
                        if dist is not None:
                            noise = torch.tensor(dist.sample(rng), device=device)
                            params_t[b] = params_t[b] + noise

            all_params.append(params_t)

            # Propagate dy0_required to next step: C1 constraint.
            # m1_t * delta_t / dur_t = v_end_t (physical ¢/s)
            # dy0_required_{t+1} = v_end_t * dur_{t+1} / delta_{t+1}
            if t + 1 < T:
                m1_t    = _norm_slope_torch(params_t[:, 0], params_t[:, 1], params_t[:, 2], at_end=True)
                v_end_t = m1_t * delta_cents[:, t] / dur_sec[:, t].clamp(min=1e-4)
                d_next  = delta_cents[:, t + 1]
                dur_next = dur_sec[:, t + 1]
                # Propagate when current AND next segment are in sta_tr_mask
                # (STA/TR and CP); reset to 0 across SIL boundaries.
                both_active = sta_tr_mask[:, t].float() * sta_tr_mask[:, t + 1].float()
                dy0_next = torch.where(
                    d_next.abs() > 1.0,
                    v_end_t * dur_next / d_next,
                    torch.zeros_like(v_end_t),
                ) * both_active
                dy0_required = dy0_next
            else:
                dy0_required = torch.zeros(B, device=device)

            slope_end_t = _slope_torch(
                params_t[:, 0], params_t[:, 1], params_t[:, 2],
                delta_cents[:, t], dur_sec[:, t], at_end=True,
            )
            all_slopes.append(slope_end_t * sta_tr_mask[:, t].float())

        return {
            "params": torch.stack(all_params, dim=1),
            "slopes": torch.stack(all_slopes, dim=1),
        }

    def decode_params(self, params_raw: torch.Tensor) -> tuple[torch.Tensor, ...]:
        log_k_raw   = params_raw[..., 0]
        logit_s_raw = params_raw[..., 1]
        A_raw       = params_raw[..., 2]
        k = torch.exp(log_k_raw * LOG_K_SCALE).clamp(K_MIN, K_MAX)
        s = torch.sigmoid(logit_s_raw * LOGIT_S_SCALE).clamp(S_MIN, S_MAX)
        A = (A_raw * A_SCALE).clamp(-A_SCALE, A_SCALE)
        return k, s, A


# ── losses ────────────────────────────────────────────────────────────────────

def slope_start_loss(
    params:            torch.Tensor,   # (B, T, 3)
    target_mask:       torch.Tensor,   # (B, T) bool
    dy0_required_norm: torch.Tensor,   # (B, T) tanh-squashed GT start slope
) -> torch.Tensor:
    """MSE between model's implied start slope and the required start slope.

    Both compared in tanh(m/M_SCALE) space for numerical stability.
    The model is trained to pick (k,s,A) that naturally produce the right
    start slope within the tanh+osc function family.
    """
    if not target_mask.any():
        return torch.tensor(0.0, device=params.device)
    m0_pred   = _norm_slope_torch(params[..., 0], params[..., 1], params[..., 2], at_end=False)
    m0_pred_n = torch.tanh(m0_pred / M_SCALE)
    return F.mse_loss(m0_pred_n[target_mask], dy0_required_norm[target_mask])


def b2_slope_loss(
    params:      torch.Tensor,
    target_mask: torch.Tensor,
    dur_sec:     torch.Tensor,
    delta_cents: torch.Tensor,
) -> torch.Tensor:
    pair_mask = target_mask[:, :-1] & target_mask[:, 1:]
    if not pair_mask.any():
        return torch.tensor(0.0, device=params.device)

    slope_end   = _slope_torch(
        params[:, :-1, 0], params[:, :-1, 1], params[:, :-1, 2],
        delta_cents[:, :-1], dur_sec[:, :-1], at_end=True,
    )
    slope_start = _slope_torch(
        params[:, 1:, 0], params[:, 1:, 1], params[:, 1:, 2],
        delta_cents[:, 1:], dur_sec[:, 1:], at_end=False,
    )
    end_n   = torch.tanh(slope_end[pair_mask]   / SLOPE_SCALE)
    start_n = torch.tanh(slope_start[pair_mask] / SLOPE_SCALE)
    return F.mse_loss(end_n, start_n)


def reconstruction_loss(
    params:      torch.Tensor,
    targets:     torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    if not target_mask.any():
        return torch.tensor(0.0, device=params.device)
    return F.smooth_l1_loss(params[target_mask], targets[target_mask])


def total_loss(
    outputs: dict,
    batch:   dict,
    cfg:     ParamGRUConfig,
) -> tuple[torch.Tensor, dict]:
    params             = outputs["params"]
    target_mask        = batch["target_mask"]
    targets            = batch["targets"]
    dy0_required_norm  = batch["dy0_required_norm"]

    recon    = reconstruction_loss(params, targets, target_mask)
    slope_s  = slope_start_loss(params, target_mask, dy0_required_norm)
    total    = recon + cfg.lambda_slope * slope_s

    return total, {
        "total_loss":       total.detach(),
        "recon_loss":       recon.detach(),
        "slope_start_loss": slope_s.detach(),
    }


# ── smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.models.param_gru.dataset_param_gru import build_dataset, collate_param_batch
    from torch.utils.data import DataLoader

    ds     = build_dataset()
    loader = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate_param_batch)
    batch  = next(iter(loader))

    cfg   = ParamGRUConfig()
    model = ParamGRU(cfg)
    out   = model(batch)
    loss, stats = total_loss(out, batch, cfg)
    print("params shape:", out["params"].shape)
    print("loss:", float(loss))
    for k, v in stats.items():
        print(f"  {k}: {float(v):.4f}")

    device = torch.device("cpu")
    dists  = fit_residuals(model, loader, device)
    print("Residual dists:", sorted(dists.keys()))
    print("pooled mean:", dists["pooled"].mean)
