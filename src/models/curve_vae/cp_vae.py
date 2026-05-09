"""
1D Convolutional VAE for CP (Charanam Position) pitch deviation curves.

Input : (batch, 1, L_CANONICAL=64) — pitch deviation in cents (mean removed)
Cond  : (batch, COND_DIM=8)        — svara one-hot (7) + log1p(dur_sec) (1)
Latent: z ∈ ℝ^LATENT_DIM (default 4)
Output: (batch, 1, 64) — reconstructed deviation curve

Architecture (symmetric encoder-decoder):
    Encoder: Conv1d ×3 (stride 2) → GlobalAvgPool → Linear → (μ, logσ²)
    Decoder: Linear → reshape → ConvTranspose1d ×3 (stride 2)

Loss: MSE_recon + β · max(KL − free_bits, 0)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import settings as S

L_CANONICAL = 64
N_SVARA     = 7
COND_DIM    = 8    # 7 svara one-hot + 1 log1p(dur)


def _is_valid_cp(curve_cents: np.ndarray, dur_sec: float) -> bool:
    """Same flatness criteria applied to GT CP regions (settings.py)."""
    if curve_cents.max() - curve_cents.min() > 2.0 * S.TOLERANCE_CENTS:
        return False
    t      = np.linspace(0.0, max(dur_sec, 1e-6), len(curve_cents))
    coeffs = np.polyfit(t, curve_cents, 1)      # [slope, intercept]
    if abs(coeffs[0]) > S.MAX_SLOPE_CENTS_PER_SEC:
        return False
    if S.MAX_RESIDUAL_CENTS is not None:
        rms = float(np.sqrt(np.mean((curve_cents - np.polyval(coeffs, t)) ** 2)))
        if rms > S.MAX_RESIDUAL_CENTS:
            return False
    return True


@dataclass
class CPVAEConfig:
    latent_dim:  int   = 4
    channels:    tuple = (16, 32, 32)   # conv channel sizes per stage
    beta:        float = 0.5
    free_bits:   float = 0.5


class CPEncoder(nn.Module):
    def __init__(self, cfg: CPVAEConfig) -> None:
        super().__init__()
        ch = cfg.channels
        self.convs = nn.Sequential(
            nn.Conv1d(1,     ch[0], kernel_size=4, stride=2, padding=1),  # 64→32
            nn.LeakyReLU(0.1),
            nn.Conv1d(ch[0], ch[1], kernel_size=4, stride=2, padding=1),  # 32→16
            nn.LeakyReLU(0.1),
            nn.Conv1d(ch[1], ch[2], kernel_size=4, stride=2, padding=1),  # 16→8
            nn.LeakyReLU(0.1),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc   = nn.Linear(ch[2] + COND_DIM, cfg.latent_dim * 2)

    def forward(
        self,
        x: torch.Tensor,      # (B, 1, L)
        cond: torch.Tensor,   # (B, COND_DIM)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.pool(self.convs(x)).squeeze(-1)   # (B, ch[-1])
        h = torch.cat([h, cond], dim=-1)
        stats = self.fc(h)
        mu, logvar = stats.chunk(2, dim=-1)
        return mu, logvar


class CPDecoder(nn.Module):
    def __init__(self, cfg: CPVAEConfig) -> None:
        super().__init__()
        ch   = cfg.channels
        # L//8 = 8 starting spatial dim
        self.fc = nn.Sequential(
            nn.Linear(cfg.latent_dim + COND_DIM, ch[2] * 8),
            nn.LeakyReLU(0.1),
        )
        self.deconvs = nn.Sequential(
            nn.ConvTranspose1d(ch[2], ch[1], kernel_size=4, stride=2, padding=1),  # 8→16
            nn.LeakyReLU(0.1),
            nn.ConvTranspose1d(ch[1], ch[0], kernel_size=4, stride=2, padding=1),  # 16→32
            nn.LeakyReLU(0.1),
            nn.ConvTranspose1d(ch[0], 1,     kernel_size=4, stride=2, padding=1),  # 32→64
        )
        self._ch2 = ch[2]

    def forward(
        self,
        z: torch.Tensor,      # (B, latent_dim)
        cond: torch.Tensor,   # (B, COND_DIM)
    ) -> torch.Tensor:
        h = self.fc(torch.cat([z, cond], dim=-1))
        h = h.view(h.size(0), self._ch2, 8)
        return self.deconvs(h)   # (B, 1, 64)


class CPVAE(nn.Module):
    def __init__(self, cfg: CPVAEConfig | None = None) -> None:
        super().__init__()
        self.cfg     = cfg or CPVAEConfig()
        self.encoder = CPEncoder(self.cfg)
        self.decoder = CPDecoder(self.cfg)

    # ------------------------------------------------------------------

    def _make_cond(
        self,
        svara_idx: torch.Tensor | None,   # (B,) long
        dur_sec:   torch.Tensor | None,   # (B,) float
        B: int,
        device: torch.device,
    ) -> torch.Tensor:
        if svara_idx is None:
            svara_oh = torch.zeros(B, N_SVARA, device=device)
        else:
            svara_oh = F.one_hot(svara_idx, num_classes=N_SVARA).float()
        if dur_sec is None:
            dur_feat = torch.zeros(B, 1, device=device)
        else:
            dur_feat = torch.log1p(dur_sec.float().to(device)).unsqueeze(-1)
        return torch.cat([svara_oh, dur_feat], dim=-1)   # (B, COND_DIM)

    def reparameterise(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        if self.training:
            std = (0.5 * logvar).exp()
            return mu + std * torch.randn_like(std)
        return mu

    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,                   # (B, 1, L)
        svara_idx: torch.Tensor | None = None,
        dur_sec:   torch.Tensor | None = None,
    ) -> dict:
        B      = x.size(0)
        device = x.device
        cond   = self._make_cond(svara_idx, dur_sec, B, device)

        mu, logvar = self.encoder(x, cond)
        z          = self.reparameterise(mu, logvar)
        recon      = self.decoder(z, cond)
        return {"recon": recon, "mu": mu, "logvar": logvar, "z": z}

    def loss(self, x: torch.Tensor, out: dict) -> dict:
        mse = F.mse_loss(out["recon"], x)
        kl  = -0.5 * (1 + out["logvar"] - out["mu"].pow(2) - out["logvar"].exp()).mean()
        kl_free = torch.clamp(kl, min=self.cfg.free_bits)
        total = mse + self.cfg.beta * kl_free
        return {"loss": total, "mse": mse, "kl": kl}

    @torch.no_grad()
    def generate(
        self,
        n:         int,
        svara_idx: torch.Tensor | None = None,
        dur_sec:   torch.Tensor | None = None,
        device:    torch.device | None = None,
        scale:     float | None        = None,
        verbose:   bool                = False,
    ) -> torch.Tensor:
        """Sample z ~ N(0,I), decode, apply Savitzky-Golay smoothing.

        If scale is given, applies rejection sampling: curves must pass the
        same flatness criteria as GT CP regions (settings.py params).
        Prints accept rate when verbose=True.

        Returns (n, 64) normalised deviation curves (multiply by scale → cents).
        """
        if device is None:
            device = next(self.parameters()).device
        self.eval()

        dur_val = float(dur_sec[0]) if (dur_sec is not None and dur_sec.numel() > 0) else 0.18

        if scale is None:
            # fast path: decode + smooth, no rejection
            z    = torch.randn(n, self.cfg.latent_dim, device=device)
            cond = self._make_cond(svara_idx, dur_sec, n, device)
            raw  = self.decoder(z, cond).squeeze(1).cpu().numpy()
            smoothed = np.stack([
                savgol_filter(c, S.CP_SAVGOL_WINDOW, S.CP_SAVGOL_POLYORDER) for c in raw
            ])
            return torch.from_numpy(smoothed.astype(np.float32))

        # rejection-sampling path
        batch_size          = max(n * 4, 64)
        accepted: list[np.ndarray] = []
        n_generated         = 0

        while len(accepted) < n:
            z = torch.randn(batch_size, self.cfg.latent_dim, device=device)
            sv_b  = svara_idx[0:1].expand(batch_size) if svara_idx is not None else None
            dur_b = dur_sec[0:1].expand(batch_size)   if dur_sec  is not None else None
            cond  = self._make_cond(sv_b, dur_b, batch_size, device)
            raw   = self.decoder(z, cond).squeeze(1).cpu().numpy()
            n_generated += batch_size

            for curve_norm in raw:
                smoothed    = savgol_filter(curve_norm, S.CP_SAVGOL_WINDOW, S.CP_SAVGOL_POLYORDER)
                curve_cents = smoothed * scale
                if _is_valid_cp(curve_cents, dur_val):
                    accepted.append(smoothed.astype(np.float32))
                    if len(accepted) >= n:
                        break

        if verbose:
            rate = n / n_generated * 100
            print(f"[cp_vae] accept {n}/{n_generated}  ({rate:.1f}%)")

        return torch.from_numpy(np.stack(accepted[:n]))
