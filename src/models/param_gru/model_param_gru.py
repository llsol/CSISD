"""
Parameter-GRU VAE: learns p(k, s, A | structural context, slope_prev, svara).

Architecture
------------
Encoder: GRU + attention pooling over full-sequence encoder input → (mu, logvar).
Decoder: GRU with z + cond at each step → (log_k_raw, logit_s_raw, A_raw) per step.

Normalization (all targets in ≈ [−1, 1]):
    log_k_raw  = log(k)  / LOG_K_SCALE    (LOG_K_SCALE = log(50) ≈ 3.91)
    logit_s_raw = logit(s) / LOGIT_S_SCALE (LOGIT_S_SCALE = 5.0)
    A_raw       = A / A_SCALE              (A_SCALE = 3.0)

B2 loss (derivative continuity):
    For consecutive STA/TR steps (i, i+1) in the same sequence,
    penalise |slope_end(i) − slope_start(i+1)|² in cents/s.
    Computed differentiably via _slope_torch().

Usage:
    from src.models.param_gru.model_param_gru import ParamGRUConfig, ParamGRUVAE
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.models.param_gru.dataset_param_gru import (
    ENC_INPUT_DIM, DEC_INPUT_DIM, OUTPUT_DIM,
    K_MIN, K_MAX, S_MIN, S_MAX,
    A_SCALE, LOG_K_SCALE, LOGIT_S_SCALE, SLOPE_SCALE,
)
from src.models.gruvae.model_gruvae import lengths_to_mask

_N_SVARA = 7


# ── config ────────────────────────────────────────────────────────────────────

@dataclass
class ParamGRUConfig:
    enc_input_dim:  int   = ENC_INPUT_DIM    # 11
    dec_input_dim:  int   = DEC_INPUT_DIM    # 9
    output_dim:     int   = OUTPUT_DIM       # 3
    hidden_dim:     int   = 64
    latent_dim:     int   = 16
    num_layers:     int   = 1
    cond_dim:       int   = 8               # svara_oh(7) + log1p(total_dur)(1)
    use_attention:  bool  = True
    beta:           float = 0.3
    free_bits:      float = 0.5
    lambda_slope:   float = 0.1             # B2 loss weight


# ── differentiable slope ──────────────────────────────────────────────────────

def _slope_torch(
    log_k_raw:   torch.Tensor,
    logit_s_raw: torch.Tensor,
    A_raw:       torch.Tensor,
    delta_cents: torch.Tensor,
    dur_sec:     torch.Tensor,
    at_end:      bool,
) -> torch.Tensor:
    """
    Slope at t=0 (at_end=False) or t=1 (at_end=True) in cents/s.
    All input tensors: (B,) or broadcastable.
    Differentiable w.r.t. log_k_raw, logit_s_raw, A_raw.
    """
    k = torch.exp(log_k_raw * LOG_K_SCALE).clamp(K_MIN, K_MAX)
    s = torch.sigmoid(logit_s_raw * LOGIT_S_SCALE).clamp(S_MIN, S_MAX)
    A = A_raw * A_SCALE

    h0    = torch.tanh(-k * s)
    h1    = torch.tanh(k * (1.0 - s))
    denom = (h1 - h0).clamp(min=1e-9)

    # h'(t) = k/cosh²(k*(t-s)) + 2πA·cos(2π*(t-0.5))
    # cos(2π*(0−0.5)) = cos(2π*(1−0.5)) = cos(±π) = −1
    if at_end:
        arg = k * (1.0 - s)
    else:
        arg = k * s    # cosh is even: cosh(k*(0-s)) = cosh(k*s)

    dh = k / torch.cosh(arg).pow(2) - 2.0 * math.pi * A
    nd = dh / denom
    return nd * delta_cents / dur_sec.clamp(min=1e-4)


# ── encoder ───────────────────────────────────────────────────────────────────

class ParamEncoder(nn.Module):
    def __init__(self, cfg: ParamGRUConfig):
        super().__init__()
        gru_drop = cfg.num_layers > 1 and cfg.num_layers * 0.0 or 0.0
        self.gru = nn.GRU(
            cfg.enc_input_dim, cfg.hidden_dim,
            num_layers=cfg.num_layers, batch_first=True,
        )
        self.use_attention = cfg.use_attention
        if cfg.use_attention:
            self.attn_q = nn.Linear(cfg.hidden_dim, 1, bias=False)
        self.to_mu     = nn.Linear(cfg.hidden_dim, cfg.latent_dim)
        self.to_logvar = nn.Linear(cfg.hidden_dim, cfg.latent_dim)

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        packed  = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, hn = self.gru(packed)

        if self.use_attention:
            out, _ = pad_packed_sequence(out, batch_first=True)  # (B, T, H)
            scores  = self.attn_q(out).squeeze(-1)               # (B, T)
            mask    = lengths_to_mask(lengths, max_len=out.size(1), device=out.device)
            scores  = scores.masked_fill(~mask, float("-inf"))
            attn_w  = torch.softmax(scores, dim=-1)
            h       = (attn_w.unsqueeze(-1) * out).sum(dim=1)
        else:
            h = hn[-1]

        return self.to_mu(h), self.to_logvar(h)


# ── decoder ───────────────────────────────────────────────────────────────────

class ParamDecoder(nn.Module):
    def __init__(self, cfg: ParamGRUConfig):
        super().__init__()
        self.cfg = cfg
        gru_in   = cfg.dec_input_dim + cfg.latent_dim + cfg.cond_dim
        self.gru = nn.GRU(gru_in, cfg.hidden_dim, num_layers=cfg.num_layers, batch_first=True)
        self.init_hidden = nn.Sequential(
            nn.Linear(cfg.latent_dim + cfg.cond_dim, cfg.hidden_dim * cfg.num_layers),
            nn.Tanh(),
        )
        self.param_head = nn.Linear(cfg.hidden_dim, cfg.output_dim)

    def _h0(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        B = z.size(0)
        h = self.init_hidden(torch.cat([z, cond], dim=-1))
        return h.view(B, self.cfg.num_layers, self.cfg.hidden_dim).transpose(0, 1).contiguous()

    def forward(
        self, z: torch.Tensor, cond: torch.Tensor,
        dec_input: torch.Tensor, lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        dec_input: (B, T, DEC_INPUT_DIM) — teacher-forced (includes GT slope_prev).
        Returns params: (B, T, 3).
        """
        B, T, _ = dec_input.shape
        z_exp    = z.unsqueeze(1).expand(B, T, -1)
        cond_exp = cond.unsqueeze(1).expand(B, T, -1)
        gru_in   = torch.cat([dec_input, z_exp, cond_exp], dim=-1)  # (B, T, gru_in_dim)

        h0     = self._h0(z, cond)
        packed = pack_padded_sequence(gru_in, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed, h0)
        out, _ = pad_packed_sequence(out, batch_first=True)         # (B, T', H)

        # pad back to T if T' < T (happens when lengths vary)
        if out.size(1) < T:
            pad = torch.zeros(B, T - out.size(1), self.cfg.hidden_dim, device=out.device)
            out = torch.cat([out, pad], dim=1)

        return self.param_head(out)  # (B, T, 3)

    def generate_step(
        self, x_t: torch.Tensor, hidden: torch.Tensor,
        z: torch.Tensor, cond: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gru_in = torch.cat([x_t, z, cond], dim=-1).unsqueeze(1)  # (B, 1, D)
        out, hidden = self.gru(gru_in, hidden)
        return self.param_head(out.squeeze(1)), hidden


# ── full model ────────────────────────────────────────────────────────────────

class ParamGRUVAE(nn.Module):
    def __init__(self, cfg: ParamGRUConfig):
        super().__init__()
        self.cfg     = cfg
        self.encoder = ParamEncoder(cfg)
        self.decoder = ParamDecoder(cfg)

    def _make_cond(self, svara_idx: torch.Tensor, total_dur: torch.Tensor) -> torch.Tensor:
        oh      = F.one_hot(svara_idx, num_classes=_N_SVARA).float()
        dur_f   = torch.log1p(total_dur.float()).unsqueeze(-1)
        return torch.cat([oh, dur_f], dim=-1)  # (B, 8)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def forward(self, batch: dict) -> dict:
        enc_input   = batch["enc_input"]
        dec_input   = batch["dec_input"]
        lengths     = batch["lengths"]
        svara_idx   = batch["svara_idx"]
        total_dur   = batch["total_dur"]

        cond = self._make_cond(svara_idx, total_dur)
        mu, logvar = self.encoder(enc_input, lengths)
        z          = self.reparameterize(mu, logvar)
        params     = self.decoder(z, cond, dec_input, lengths)

        return {"params": params, "mu": mu, "logvar": logvar, "z": z}

    def generate(
        self,
        seg_type_oh: torch.Tensor,   # (B, T, 6)
        dur_rel:     torch.Tensor,   # (B, T)
        delta_norm:  torch.Tensor,   # (B, T)
        dur_sec:     torch.Tensor,   # (B, T)
        delta_cents: torch.Tensor,   # (B, T)
        sta_tr_mask: torch.Tensor,   # (B, T) bool — True for STA/TR positions
        lengths:     torch.Tensor,   # (B,)
        svara_idx:   torch.Tensor,   # (B,)
        total_dur:   torch.Tensor,   # (B,)
        z:           torch.Tensor | None = None,
    ) -> dict:
        """
        Autoregressive generation: slope_prev propagated from predicted params.
        Returns params (B, T, 3) and slopes (B, T).
        """
        B, T, _  = seg_type_oh.shape
        device   = seg_type_oh.device

        if z is None:
            z = torch.randn(B, self.cfg.latent_dim, device=device)
        cond    = self._make_cond(svara_idx.to(device), total_dur.to(device))
        hidden  = self.decoder._h0(z, cond)

        slope_prev = torch.zeros(B, device=device)
        all_params = []
        all_slopes = []

        for t in range(T):
            slope_prev_n = torch.tanh(slope_prev / SLOPE_SCALE).unsqueeze(-1)  # (B,1)
            x_t = torch.cat([
                seg_type_oh[:, t, :],
                dur_rel[:, t].unsqueeze(-1),
                delta_norm[:, t].unsqueeze(-1),
                slope_prev_n,
            ], dim=-1)  # (B, DEC_INPUT_DIM)

            params_t, hidden = self.decoder.generate_step(x_t, hidden, z, cond)
            all_params.append(params_t)

            # compute slope_end from predicted params (for STA/TR)
            slope_end_t = _slope_torch(
                params_t[:, 0], params_t[:, 1], params_t[:, 2],
                delta_cents[:, t], dur_sec[:, t], at_end=True,
            )
            # For non-STA/TR positions, keep slope_prev = 0
            is_sta_tr = sta_tr_mask[:, t].float()
            slope_prev = slope_end_t * is_sta_tr
            all_slopes.append(slope_prev)

        return {
            "params": torch.stack(all_params, dim=1),    # (B, T, 3)
            "slopes": torch.stack(all_slopes, dim=1),    # (B, T)
        }

    def decode_params(self, params_raw: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Convert raw model output to (k, s, A). params_raw: (..., 3)."""
        log_k_raw   = params_raw[..., 0]
        logit_s_raw = params_raw[..., 1]
        A_raw       = params_raw[..., 2]
        k = torch.exp(log_k_raw * LOG_K_SCALE).clamp(K_MIN, K_MAX)
        s = torch.sigmoid(logit_s_raw * LOGIT_S_SCALE).clamp(S_MIN, S_MAX)
        A = (A_raw * A_SCALE).clamp(-A_SCALE, A_SCALE)
        return k, s, A


# ── losses ────────────────────────────────────────────────────────────────────

def kl_loss(mu: torch.Tensor, logvar: torch.Tensor, free_bits: float = 0.0) -> torch.Tensor:
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    if free_bits > 0.0:
        kl_per_dim = kl_per_dim.clamp(min=free_bits)
    return kl_per_dim.sum(dim=1).mean()


def b2_slope_loss(
    params:     torch.Tensor,    # (B, T, 3)
    target_mask: torch.Tensor,   # (B, T) bool — True for STA/TR
    dur_sec:    torch.Tensor,    # (B, T)
    delta_cents: torch.Tensor,   # (B, T)
) -> torch.Tensor:
    """
    Penalise slope discontinuity at consecutive STA/TR boundaries.
    pair_mask[b, t] = True if positions t AND t+1 are both STA/TR.
    """
    pair_mask = target_mask[:, :-1] & target_mask[:, 1:]   # (B, T-1)
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
    # Normalise with tanh so the loss stays in [0, 4] regardless of scale
    end_n   = torch.tanh(slope_end[pair_mask]   / SLOPE_SCALE)
    start_n = torch.tanh(slope_start[pair_mask] / SLOPE_SCALE)
    return F.mse_loss(end_n, start_n)


def reconstruction_loss(
    params:      torch.Tensor,   # (B, T, 3) model output
    targets:     torch.Tensor,   # (B, T, 3) GT
    target_mask: torch.Tensor,   # (B, T) bool
) -> torch.Tensor:
    """Smooth-L1 loss on (log_k_raw, logit_s_raw, A_raw) for STA/TR only."""
    if not target_mask.any():
        return torch.tensor(0.0, device=params.device)
    return F.smooth_l1_loss(params[target_mask], targets[target_mask])


def total_loss(
    outputs:     dict,
    batch:       dict,
    cfg:         ParamGRUConfig,
    beta:        float | None = None,
) -> tuple[torch.Tensor, dict]:
    if beta is None:
        beta = cfg.beta

    params       = outputs["params"]
    mu, logvar   = outputs["mu"], outputs["logvar"]
    target_mask  = batch["target_mask"]
    targets      = batch["targets"]
    dur_sec      = batch["dur_sec"]
    delta_cents  = batch["delta_cents"]

    recon  = reconstruction_loss(params, targets, target_mask)
    kl     = kl_loss(mu, logvar, free_bits=cfg.free_bits)
    b2     = b2_slope_loss(params, target_mask, dur_sec, delta_cents)
    total  = recon + beta * kl + cfg.lambda_slope * b2

    return total, {
        "total_loss": total.detach(),
        "recon_loss": recon.detach(),
        "kl_loss":    kl.detach(),
        "b2_loss":    b2.detach(),
        "beta":       torch.tensor(beta),
    }


# ── smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.models.param_gru.dataset_param_gru import build_dataset, collate_param_batch
    from torch.utils.data import DataLoader

    ds     = build_dataset()
    loader = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate_param_batch)
    batch  = next(iter(loader))

    cfg    = ParamGRUConfig()
    model  = ParamGRUVAE(cfg)
    out    = model(batch)
    loss, stats = total_loss(out, batch, cfg)
    print("Batch keys:", list(batch.keys()))
    print("params shape:", out["params"].shape)
    print("loss:", float(loss))
    for k, v in stats.items():
        print(f"  {k}: {float(v):.4f}")
