"""
GRU + VAE model for structural svara representation.

Input per segment (MODEL_INPUT_DIM = 6):
    [onehot_CP, onehot_SIL, onehot_STA, onehot_TR, dur_rel, cents_norm]

Note: dataset_gruvae produces sequences with INPUT_DIM = 7, which includes
total_dur_sec at index 5. Use DATASET_FEATURE_COLS to slice the right features
before feeding to the model:

    x_model = x_dataset[:, :, DATASET_FEATURE_COLS]   # (batch, seq_len, 6)

Shape conventions:
    x:        (batch, seq_len, MODEL_INPUT_DIM)
    lengths:  (batch,)                            — real segment count per svara

Length predictor
----------------
A dedicated linear head predicts n_segments from z (after reparameterisation).
This forces the encoder to encode sequence length into the latent space and
lets generate() use the predicted length instead of a fixed max_seq_len.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# Indices to select from dataset_gruvae's 7-feature sequences
# dataset cols: [oh_CP, oh_SIL, oh_STA, oh_TR, dur_rel, total_dur_sec, cents_norm]
# model cols:   [oh_CP, oh_SIL, oh_STA, oh_TR, dur_rel,                cents_norm]
DATASET_FEATURE_COLS = [0, 1, 2, 3, 4, 6]
MODEL_INPUT_DIM = 6   # == len(DATASET_FEATURE_COLS)

# Type indices: CP=0, SIL=1, STA=2, TR=3
# Musical grammar: which types can follow each type.
# TR→{CP} only; TR at end-of-sequence is fine (no successor needed).
VALID_NEXT: dict[int, set[int]] = {
    0: {1, 2, 3},     # CP  → SIL, STA, TR  (no CP→CP: force break after stable region)
    1: {0, 2, 3},     # SIL → CP, STA, TR
    2: {1, 2, 3},     # STA → SIL, STA, TR  (STA→CP forbidden: always via TR by definition)
    3: {0},           # TR  → CP only
}

# Number of svara labels — always 7, constant across config variants
_N_SVARA = 7


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

@dataclass
class ModelConfig:
    input_dim:              int   = MODEL_INPUT_DIM
    type_dim:               int   = 4
    hidden_dim:             int   = 128
    latent_dim:             int   = 16
    num_layers:             int   = 1
    dropout:                float = 0.0
    max_seq_len:            int   = 32
    condition_z_every_step: bool  = True
    teacher_forcing_ratio:  float = 1.0

    # Conditional VAE: condition decoder on svara label one-hot + log1p(total_dur_sec).
    # 7 = svara one-hot only (backward compat); 8 = svara one-hot + duration scalar.
    # Set 0 to disable conditioning entirely.
    svara_cond_dim:         int   = 7

    lambda_type:            float = 1.0
    lambda_dur_cp:          float = 0.3   # CP dur — moderately invariant (MI_perf≈0.05)
    lambda_dur_sta:         float = 0.05  # STA dur — performer-dependent (MI_perf≈0.26-0.39)
    lambda_dur_sil:         float = 0.1   # SIL dur — weak signal
    lambda_cp_cents:        float = 2.0   # CP pitch — discriminative but performer-dependent
    lambda_sta_cents:       float = 5.0   # STA pitch — peak value, most musicologically relevant
    lambda_length:          float = 0.1   # weight for n_segments prediction loss
    lambda_dur_tr:          float = 0.135   # TR duration — return time, informative

    use_attention:          bool  = True   # pool all GRU states via learned attention

    beta:                   float = 1.0
    free_bits:              float = 0.0   # min nats per latent dim; 0 = disabled
    use_huber_for_continuous: bool = True


# ------------------------------------------------------------
# UTILITIES
# ------------------------------------------------------------

def lengths_to_mask(
    lengths: torch.Tensor,
    max_len: int | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Boolean mask (batch, max_len): True where position < length."""
    if max_len is None:
        max_len = int(lengths.max().item())
    if device is None:
        device = lengths.device
    positions = torch.arange(max_len, device=device).unsqueeze(0)
    return positions < lengths.unsqueeze(1)


def onehot_to_index(type_onehot: torch.Tensor) -> torch.Tensor:
    return type_onehot.argmax(dim=-1)


def build_start_token(batch_size: int, input_dim: int, device: torch.device) -> torch.Tensor:
    return torch.zeros(batch_size, input_dim, device=device)


# ------------------------------------------------------------
# ENCODER
# ------------------------------------------------------------

class SvaraEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        gru_dropout = cfg.dropout if cfg.num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=cfg.input_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.use_attention = cfg.use_attention
        if cfg.use_attention:
            self.attn_q = nn.Linear(cfg.hidden_dim, 1, bias=False)
        self.to_mu     = nn.Linear(cfg.hidden_dim, cfg.latent_dim)
        self.to_logvar = nn.Linear(cfg.hidden_dim, cfg.latent_dim)

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        outputs, h_n = self.gru(packed)

        if self.use_attention:
            outputs, _ = pad_packed_sequence(outputs, batch_first=True)  # (B, T, H)
            scores = self.attn_q(outputs).squeeze(-1)                    # (B, T)
            mask   = lengths_to_mask(lengths, max_len=outputs.size(1), device=outputs.device)
            scores = scores.masked_fill(~mask, float("-inf"))
            attn_w = torch.softmax(scores, dim=-1)                       # (B, T)
            h      = (attn_w.unsqueeze(-1) * outputs).sum(dim=1)        # (B, H)
        else:
            attn_w = None
            h      = h_n[-1]

        return self.to_mu(h), self.to_logvar(h), attn_w


# ------------------------------------------------------------
# DECODER
# ------------------------------------------------------------

class SvaraDecoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        cond_dim = cfg.svara_cond_dim
        decoder_input_dim = (
            cfg.input_dim
            + (cfg.latent_dim if cfg.condition_z_every_step else 0)
            + cond_dim
        )

        gru_dropout = cfg.dropout if cfg.num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=decoder_input_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        # Map z (+ condition) to initial hidden state; tanh keeps values in GRU-friendly range
        self.init_hidden = nn.Sequential(
            nn.Linear(cfg.latent_dim + cond_dim, cfg.hidden_dim * cfg.num_layers),
            nn.Tanh(),
        )

        self.type_head  = nn.Linear(cfg.hidden_dim, cfg.type_dim)
        self.dur_head   = nn.Linear(cfg.hidden_dim, 1)
        self.cents_head = nn.Linear(cfg.hidden_dim, 1)

    def latent_to_hidden(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        batch_size = z.size(0)
        hidden = self.init_hidden(torch.cat([z, cond], dim=-1))
        hidden = hidden.view(batch_size, self.cfg.num_layers, self.cfg.hidden_dim)
        return hidden.transpose(0, 1).contiguous()

    def step(self, prev_input, hidden, z, cond):
        parts = [prev_input]
        if self.cfg.condition_z_every_step:
            parts.append(z)
        if self.cfg.svara_cond_dim > 0:
            parts.append(cond)
        step_input = torch.cat(parts, dim=-1)

        output, hidden = self.gru(step_input.unsqueeze(1), hidden)
        output = output.squeeze(1)

        type_logits = self.type_head(output)
        dur_pred    = self.dur_head(output).squeeze(-1)
        cents_pred  = self.cents_head(output).squeeze(-1)
        return type_logits, dur_pred, cents_pred, hidden

    def _next_input_from_preds(
        self,
        type_logits: torch.Tensor,
        dur_pred: torch.Tensor,
        cents_pred: torch.Tensor,
    ) -> torch.Tensor:
        idx = type_logits.argmax(dim=-1)
        oh  = F.one_hot(idx, num_classes=self.cfg.type_dim).float()
        # SIL (idx=1) and TR (idx=3) have cents=0 by definition — enforce this
        # so the auto-regressive input matches what the model saw during training.
        no_cents_mask = ((idx == 1) | (idx == 3)).float()
        cents_in = cents_pred * (1.0 - no_cents_mask)
        return torch.cat([oh, dur_pred.unsqueeze(-1), cents_in.unsqueeze(-1)], dim=-1)

    def forward(
        self,
        z: torch.Tensor,
        cond: torch.Tensor,
        target_seq: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 1.0,
        max_len: int | None = None,
        use_hard_mask: bool = False,
    ) -> dict:
        batch_size = z.size(0)
        device     = z.device
        hidden     = self.latent_to_hidden(z, cond)

        if max_len is None:
            max_len = target_seq.size(1) if target_seq is not None else self.cfg.max_seq_len

        prev_input = build_start_token(batch_size, self.cfg.input_dim, device)

        type_logits_all, dur_all, cents_all = [], [], []
        # Track previous predicted type index per sample (None = start token, no constraint)
        prev_type_idx: torch.Tensor | None = None

        # Build per-type valid-next masks once (on CPU, then move to device)
        if use_hard_mask:
            n_types = self.cfg.type_dim
            # invalid_mask[prev_type, next_type] = True means BLOCK next_type
            invalid_mask = torch.ones(n_types, n_types, dtype=torch.bool, device=device)
            for prev_t, valid_set in VALID_NEXT.items():
                for nxt in valid_set:
                    invalid_mask[prev_t, nxt] = False

        for t in range(max_len):
            type_logits, dur_pred, cents_pred, hidden = self.step(prev_input, hidden, z, cond)

            # Apply hard grammar mask based on previous predicted type
            if use_hard_mask and prev_type_idx is not None:
                # prev_type_idx: (batch,)  →  row into invalid_mask
                per_sample_mask = invalid_mask[prev_type_idx]          # (batch, n_types)
                type_logits = type_logits.masked_fill(per_sample_mask, float("-inf"))

            type_logits_all.append(type_logits)
            dur_all.append(dur_pred)
            cents_all.append(cents_pred)

            use_teacher = (
                self.training
                and target_seq is not None
                and torch.rand(1, device=device).item() < teacher_forcing_ratio
            )
            if use_teacher:
                prev_input     = target_seq[:, t, :]
                prev_type_idx  = target_seq[:, t, :self.cfg.type_dim].argmax(dim=-1)
            else:
                prev_input     = self._next_input_from_preds(type_logits, dur_pred, cents_pred)
                prev_type_idx  = type_logits.argmax(dim=-1)

        return {
            "type_logits": torch.stack(type_logits_all, dim=1),
            "duration":    torch.stack(dur_all, dim=1),
            "cents":       torch.stack(cents_all, dim=1),
        }


# ------------------------------------------------------------
# FULL MODEL
# ------------------------------------------------------------

class SvaraGRUVAE(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg     = cfg
        self.encoder = SvaraEncoder(cfg)
        self.decoder = SvaraDecoder(cfg)
        # Predicts n_segments from z; trained jointly via lambda_length loss
        self.length_head = nn.Linear(cfg.latent_dim, 1)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def predict_length(self, z: torch.Tensor) -> torch.Tensor:
        """
        Predict n_segments from z. Shape: (batch,).
        Internally operates on normalised scale (n / max_seq_len) so the loss
        stays in [0, 1] and is comparable to the other continuous losses.
        """
        norm = torch.sigmoid(self.length_head(z).squeeze(-1))   # (0, 1)
        return (norm * self.cfg.max_seq_len).clamp(min=1.0)

    def _make_cond(
        self,
        svara_idx: torch.Tensor | None,
        total_dur: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Condition vector (batch, svara_cond_dim).
        svara_cond_dim == 0: empty (unconditional).
        svara_cond_dim == 7: svara one-hot only.
        svara_cond_dim == 8: svara one-hot + log1p(total_dur_sec).
        """
        if self.cfg.svara_cond_dim == 0:
            return torch.zeros(batch_size, 0, device=device)
        if svara_idx is None:
            svara_oh = torch.zeros(batch_size, _N_SVARA, device=device)
        else:
            svara_oh = F.one_hot(svara_idx, num_classes=_N_SVARA).float()
        if self.cfg.svara_cond_dim <= _N_SVARA:
            return svara_oh
        if total_dur is None:
            dur_feat = torch.zeros(batch_size, 1, device=device)
        else:
            dur_feat = torch.log1p(total_dur.float().to(device)).unsqueeze(-1)
        return torch.cat([svara_oh, dur_feat], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        svara_idx: torch.Tensor | None = None,
        total_dur: torch.Tensor | None = None,
        teacher_forcing_ratio: float | None = None,
    ) -> dict:
        if teacher_forcing_ratio is None:
            teacher_forcing_ratio = self.cfg.teacher_forcing_ratio

        mu, logvar, attn_w = self.encoder(x, lengths)
        z = self.reparameterize(mu, logvar)
        cond = self._make_cond(svara_idx, total_dur, x.size(0), x.device)
        decoded = self.decoder(
            z, cond, target_seq=x,
            teacher_forcing_ratio=teacher_forcing_ratio,
            max_len=x.size(1),
        )
        return {
            "mu": mu, "logvar": logvar, "z": z, "attn_w": attn_w,
            "pred_length": self.predict_length(z),
            **decoded,
        }

    def generate(
        self,
        batch_size: int = 1,
        z: torch.Tensor | None = None,
        svara_idx: torch.Tensor | None = None,
        total_dur: torch.Tensor | None = None,
        max_len: int | None = None,
        device: torch.device | None = None,
        use_hard_mask: bool = False,
    ) -> dict:
        if device is None:
            device = next(self.parameters()).device
        if z is None:
            z = torch.randn(batch_size, self.cfg.latent_dim, device=device)

        cond = self._make_cond(svara_idx, total_dur, z.size(0), device)

        self.eval()
        with torch.no_grad():
            if max_len is None:
                pred_len = self.predict_length(z)
                max_len  = int(pred_len.round().clamp(1, self.cfg.max_seq_len).max().item())

            decoded = self.decoder(
                z, cond, teacher_forcing_ratio=0.0, max_len=max_len,
                use_hard_mask=use_hard_mask,
            )
            idx = decoded["type_logits"].argmax(dim=-1)   # (batch, seq_len)

            # Final-state constraint: STA (idx=2) cannot be the last segment.
            # Replace terminal STA with TR (idx=3).
            if use_hard_mask:
                last_positions = (
                    pred_len.round().clamp(1, max_len).long() - 1
                )  # (batch,)
                for b in range(idx.size(0)):
                    last_pos = int(last_positions[b].item())
                    if idx[b, last_pos].item() == 2:   # STA
                        idx[b, last_pos] = 3            # → TR

            oh  = F.one_hot(idx, num_classes=self.cfg.type_dim).float()
            generated = torch.cat(
                [oh, decoded["duration"].unsqueeze(-1), decoded["cents"].unsqueeze(-1)],
                dim=-1,
            )
        return {"z": z, "generated": generated, "pred_length": pred_len, **decoded}


# ------------------------------------------------------------
# LOSSES
# ------------------------------------------------------------

def reconstruction_loss(
    outputs: dict,
    targets: torch.Tensor,
    lengths: torch.Tensor,
    cfg: ModelConfig,
) -> tuple[torch.Tensor, dict]:
    """
    targets: (batch, seq_len, MODEL_INPUT_DIM=6)
        [:, :, :4]  one-hot type  [CP, SIL, STA, TR]
        [:, :,  4]  dur_rel
        [:, :,  5]  cents_norm
    """
    device = targets.device
    batch_size, seq_len, _ = targets.shape
    mask = lengths_to_mask(lengths, max_len=seq_len, device=device).float()

    target_type_idx = onehot_to_index(targets[:, :, :cfg.type_dim])
    target_duration = targets[:, :, cfg.type_dim]        # index 3
    target_cents    = targets[:, :, cfg.type_dim + 1]    # index 4

    type_loss_all = F.cross_entropy(
        outputs["type_logits"].reshape(-1, cfg.type_dim),
        target_type_idx.reshape(-1),
        reduction="none",
    ).view(batch_size, seq_len)

    if cfg.use_huber_for_continuous:
        dur_loss_all   = F.smooth_l1_loss(outputs["duration"], target_duration, reduction="none")
        cents_loss_all = F.smooth_l1_loss(outputs["cents"],    target_cents,    reduction="none")
    else:
        dur_loss_all   = F.mse_loss(outputs["duration"], target_duration, reduction="none")
        cents_loss_all = F.mse_loss(outputs["cents"],    target_cents,    reduction="none")

    cp_mask  = targets[:, :, 0]   # (B, T)
    sil_mask = targets[:, :, 1]   # (B, T)
    sta_mask = targets[:, :, 2]   # (B, T)
    tr_mask  = targets[:, :, 3]   # (B, T)

    dur_weight = (
        cfg.lambda_dur_cp  * cp_mask  +
        cfg.lambda_dur_sta * sta_mask +
        cfg.lambda_dur_sil * sil_mask +
        cfg.lambda_dur_tr  * tr_mask
    )
    # TR and SIL have cents=0, weight=0 → no gradient for cents on those types
    cents_weight = (
        cfg.lambda_cp_cents  * cp_mask  +
        cfg.lambda_sta_cents * sta_mask
    )

    n = mask.sum().clamp_min(1.0)
    type_loss  = (type_loss_all  * mask).sum() / n
    dur_loss   = (dur_loss_all   * dur_weight * mask).sum() / n
    cents_loss = (cents_loss_all * cents_weight * mask).sum() / n

    # Length prediction loss — computed on normalised scale [0, 1]
    length_loss = F.smooth_l1_loss(
        outputs["pred_length"] / cfg.max_seq_len,
        lengths.float()        / cfg.max_seq_len,
    )

    loss = cfg.lambda_type * type_loss + dur_loss + cents_loss + cfg.lambda_length * length_loss

    return loss, {
        "type_loss":   type_loss.detach(),
        "dur_loss":    dur_loss.detach(),
        "cents_loss":  cents_loss.detach(),
        "length_loss": length_loss.detach(),
    }


def kl_loss(mu: torch.Tensor, logvar: torch.Tensor, free_bits: float = 0.0) -> torch.Tensor:
    """
    KL divergence with optional free bits per latent dimension.

    free_bits > 0: each dimension is not penalised below `free_bits` nats,
    which forces the encoder to keep KL >= free_bits * latent_dim and prevents
    full posterior collapse.
    """
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # (batch, latent_dim)
    if free_bits > 0.0:
        kl_per_dim = kl_per_dim.clamp(min=free_bits)
    return kl_per_dim.sum(dim=1).mean()


def total_vae_loss(
    outputs: dict,
    targets: torch.Tensor,
    lengths: torch.Tensor,
    cfg: ModelConfig,
    beta: float | None = None,
) -> tuple[torch.Tensor, dict]:
    if beta is None:
        beta = cfg.beta
    recon, recon_parts = reconstruction_loss(outputs, targets, lengths, cfg)
    kl = kl_loss(outputs["mu"], outputs["logvar"], free_bits=cfg.free_bits)
    total = recon + beta * kl
    return total, {
        "total_loss": total.detach(),
        "recon_loss": recon.detach(),
        "kl_loss":    kl.detach(),
        **recon_parts,
    }


# ------------------------------------------------------------
# KL ANNEALING
# ------------------------------------------------------------

def linear_kl_beta(step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, step / float(warmup_steps))


# ------------------------------------------------------------
# TRAIN STEP
# ------------------------------------------------------------

def train_step(
    model: SvaraGRUVAE,
    batch_x: torch.Tensor,
    batch_lengths: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    warmup_steps: int = 1000,
    total_dur: torch.Tensor | None = None,
) -> dict:
    model.train()
    optimizer.zero_grad()
    beta = linear_kl_beta(global_step, warmup_steps)
    outputs = model(batch_x, batch_lengths, total_dur=total_dur)
    loss, stats = total_vae_loss(outputs, batch_x, batch_lengths, model.cfg, beta=beta)
    loss.backward()
    optimizer.step()
    stats["beta"] = torch.tensor(beta, device=batch_x.device)
    return stats


# ------------------------------------------------------------
# SMOKE TEST
# ------------------------------------------------------------

if __name__ == "__main__":
    cfg   = ModelConfig()
    model = SvaraGRUVAE(cfg)

    B, T = 4, 10
    x = torch.zeros(B, T, cfg.input_dim)
    types = torch.randint(0, cfg.type_dim, (B, T))
    x[:, :, :cfg.type_dim]    = F.one_hot(types, num_classes=cfg.type_dim).float()
    x[:, :, cfg.type_dim]     = torch.rand(B, T)            # dur_rel
    x[:, :, cfg.type_dim + 1] = torch.randn(B, T) * 0.3     # cents_norm

    lengths   = torch.tensor([10, 8, 6, 9])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    stats = train_step(model, x, lengths, optimizer, global_step=100)
    print("Train stats:")
    for k, v in stats.items():
        print(f"  {k}: {float(v):.4f}")

    out = model.generate(batch_size=2)
    print("Generated shape:", out["generated"].shape)
    print("Predicted lengths:", out["pred_length"].tolist())
