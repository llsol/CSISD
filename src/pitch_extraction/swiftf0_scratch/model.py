"""
SwiftF0-scratch: SwiftF0 architecture trained from scratch on SCMS Carnatic.

Differences from swiftf0_finetune/model.py:
    N_FFT      1024  →  2048   (STFT resolution: ~62 ¢/bin → ~31 ¢/bin at 300 Hz)
    N_FREQ_BINS 132  →   262   (STFT bins 6:268 instead of 3:135)
    N_PITCH_BINS 200 →   360   (bin width: ~33 ¢ → ~18 ¢)
    Weights        ONNX preentrenat  →  inicialització aleatòria PyTorch

Architecture (identical to SwiftF0, only dimensions change):
    audio (16 kHz, padded 896 samples each side)
        → STFT (n_fft=2048, hop=256, onesided)
        → magnitude[:, 6:268, :]  → log(mag + ε)
        → log_spec (B, 262, T)

    log_spec.unsqueeze(1)
        → Conv2d stack (1→8→16→32→64→1, 5×5, pad=2, all ReLU)
        → squeeze(1)                       (B, 262, T)
        → freq_projection Conv1d(262→360, k=1)
        → softmax over 360 bins            (B, T, 360)
        → pitch_hz, confidence, pitch_probs
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# ── STFT / frequency constants ───────────────────────────────────────────────
SR            = 16_000
N_FFT         = 2_048
HOP           = 256
PAD           = (N_FFT - HOP) // 2   # 896 — symmetric padding before STFT
FREQ_LOW      = 6                     # bin ≥ 46.875 Hz  (6 × 7.8125)
FREQ_HIGH     = 268                   # bin ≤ 2093.75 Hz (268 × 7.8125)
N_FREQ_BINS   = FREQ_HIGH - FREQ_LOW  # 262
N_PITCH_BINS  = 360                   # ~18.3 ¢/bin (log-uniform, same Hz range)
CONF_HALF_WIN = 9                     # ±bins around argmax for confidence
FMIN          = 46.875                # Hz — lower bound (same as SwiftF0)
FMAX          = 2_093.75             # Hz — upper bound (same as SwiftF0)
# ─────────────────────────────────────────────────────────────────────────────


def _compute_bin_centers(n_bins: int = N_PITCH_BINS) -> torch.Tensor:
    """360 log-uniform bin centres between FMIN and FMAX (Hz)."""
    centers = np.exp(np.linspace(np.log(FMIN), np.log(FMAX), n_bins))
    return torch.from_numpy(centers.astype(np.float32))


class SwiftF0Scratch(nn.Module):
    """
    SwiftF0 architecture trained from scratch on Carnatic SCMS data.

    forward(audio) → (pitch_hz, confidence, pitch_probs) where:
        pitch_hz    (B, T)       Hz, weighted average within ±9 bins of peak
        confidence  (B, T)       [0, 1] — prob mass within ±9 bins
        pitch_probs (B, 360, T)  softmax distribution; used for CE loss
    """

    def __init__(self):
        super().__init__()

        self.register_buffer("window", torch.hann_window(N_FFT))

        # Conv stack — identical to SwiftF0, operates on 262 freq bins
        self.conv1     = nn.Conv2d(1,  8,  5, padding=2)
        self.conv2     = nn.Conv2d(8,  16, 5, padding=2)
        self.conv3     = nn.Conv2d(16, 32, 5, padding=2)
        self.conv4     = nn.Conv2d(32, 64, 5, padding=2)
        self.conf_head = nn.Conv2d(64,  1, 5, padding=2)

        # Pitch projection: 262 enhanced freq features → 360 pitch bins
        self.freq_projection = nn.Conv1d(N_FREQ_BINS, N_PITCH_BINS, 1)

        # Fixed lookup tables
        self.register_buffer("pitch_bin_centers", _compute_bin_centers())
        self.register_buffer(
            "bin_indices",
            torch.arange(N_PITCH_BINS, dtype=torch.float32),
        )

    # ── forward ──────────────────────────────────────────────────────────────

    def _log_spec(self, audio: torch.Tensor) -> torch.Tensor:
        """audio (B, L) → log-magnitude slice (B, 262, T)."""
        audio = F.pad(audio, (PAD, PAD))
        stft = torch.stft(
            audio,
            n_fft=N_FFT,
            hop_length=HOP,
            win_length=N_FFT,
            window=self.window,
            center=False,
            return_complex=True,
        )                                           # (B, 1025, T)
        mag = stft.abs()[:, FREQ_LOW:FREQ_HIGH, :] # (B, 262, T)
        return torch.log(mag + 1e-9)

    def _softmax_to_pitch_conf(
        self, probs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        probs: (B, T, 360)
        Returns pitch_hz (B, T) and confidence (B, T).
        """
        B, T, _ = probs.shape
        p_flat = probs.reshape(B * T, N_PITCH_BINS)

        argmax = p_flat.argmax(dim=-1)
        dist   = (self.bin_indices.unsqueeze(0) -
                  argmax.float().unsqueeze(1)).abs()
        mask   = (dist <= CONF_HALF_WIN).float()

        masked    = p_flat * mask
        conf_flat = masked.sum(dim=-1)
        norm_flat = masked / (conf_flat.unsqueeze(1) + 1e-7)

        centers    = self.pitch_bin_centers.unsqueeze(0)
        pitch_flat = (norm_flat * centers).sum(dim=-1)

        return pitch_flat.reshape(B, T), conf_flat.reshape(B, T)

    def forward(self, audio: torch.Tensor):
        """
        audio : (B, L) float32 at 16 kHz
        Returns:
            pitch_hz    (B, T)
            confidence  (B, T)
            pitch_probs (B, 360, T)
        """
        log_spec = self._log_spec(audio)           # (B, 262, T)

        x = log_spec.unsqueeze(1)                  # (B, 1, 262, T)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conf_head(x))              # (B, 1, 262, T)
        x = x.squeeze(1)                           # (B, 262, T)

        logits = self.freq_projection(x)           # (B, 360, T)
        probs  = F.softmax(logits.permute(0, 2, 1), dim=-1)  # (B, T, 360)

        pitch_hz, confidence = self._softmax_to_pitch_conf(probs)
        pitch_probs = probs.permute(0, 2, 1)       # (B, 360, T) for CE loss

        return pitch_hz, confidence, pitch_probs

    # ── export ───────────────────────────────────────────────────────────────

    def export_onnx(self, out_path: str | Path, audio_len: int = 16_000 * 30):
        self.eval()
        dummy = torch.zeros(1, audio_len)
        torch.onnx.export(
            self,
            dummy,
            str(out_path),
            input_names=["input_audio"],
            output_names=["pitch_hz", "confidence", "pitch_probs"],
            dynamic_axes={"input_audio": {1: "audio_length"}},
            opset_version=17,
        )
        print(f"[SwiftF0Scratch] Exported ONNX → {out_path}")
