"""
SwiftF0 PyTorch reimplementation for fine-tuning.

Architecture (reverse-engineered from model.onnx, 96 nodes):

    audio (16 kHz, padded 384 samples each side)
        → STFT (n_fft=1024, hop=256, onesided)
        → magnitude = sqrt(real² + imag²)
        → slice bins [3:135] → log(mag + ε)
        → log_spec (B, 132, T)

    Single branch:
        log_spec.unsqueeze(1) → Conv2d stack (1→8→16→32→64→1, 5×5, pad=2, all ReLU)
        → squeeze(dim=1) → freq_projection Conv1d(132→200, k=1)
        → softmax over 200 pitch bins per frame
        → pitch_probs (B, T, 200)

    Confidence (inference): sum of probabilities within ±9 bins of argmax
    Confidence (training):  max prob per frame — differentiable proxy

Weights are loaded from the installed swift_f0 ONNX model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── STFT / frequency constants (match swift_f0 package) ─────────────────────
SR           = 16_000
N_FFT        = 1_024
HOP          = 256
PAD          = 384          # samples padded on each side before STFT
FREQ_LOW     = 3            # first bin ≥ 46.875 Hz
FREQ_HIGH    = 135          # last bin  ≤ 2093.75 Hz
N_FREQ_BINS  = FREQ_HIGH - FREQ_LOW  # 132
N_PITCH_BINS = 200
CONF_HALF_WIN = 9           # ±bins around argmax for confidence computation
# ────────────────────────────────────────────────────────────────────────────


def _onnx_weights(onnx_path: str | Path) -> dict[str, np.ndarray]:
    m = onnx.load(str(onnx_path))
    return {init.name: numpy_helper.to_array(init) for init in m.graph.initializer}


def _default_onnx_path() -> Path:
    import swift_f0
    return Path(swift_f0.__file__).parent / "model.onnx"


class SwiftF0(nn.Module):
    """
    Trainable PyTorch version of SwiftF0.

    forward(audio) returns (pitch_hz, confidence, pitch_probs) where:
        pitch_hz    (B, T)      Hz, soft weighted average within ±9 bins
        confidence  (B, T)      [0, 1] — probability mass within ±9 bins of peak
        pitch_probs (B, 200, T) Softmax distribution; used for training CE loss
    """

    def __init__(self):
        super().__init__()

        self.register_buffer("window", torch.hann_window(N_FFT))

        # Single conv stack — all layers feed into freq_projection
        self.conv1     = nn.Conv2d(1,  8,  5, padding=2)
        self.conv2     = nn.Conv2d(8,  16, 5, padding=2)
        self.conv3     = nn.Conv2d(16, 32, 5, padding=2)
        self.conv4     = nn.Conv2d(32, 64, 5, padding=2)
        self.conf_head = nn.Conv2d(64,  1, 5, padding=2)

        # Pitch projection: maps 132 enhanced freq features → 200 pitch bins
        self.freq_projection = nn.Conv1d(N_FREQ_BINS, N_PITCH_BINS, 1)

        # Pitch bin centre frequencies (Hz) — fixed lookup table
        self.register_buffer("pitch_bin_centers", torch.zeros(N_PITCH_BINS))

        # Fixed bin-index tensor for windowed confidence/pitch computation
        self.register_buffer(
            "bin_indices",
            torch.arange(N_PITCH_BINS, dtype=torch.float32),
        )

    # ── weight loading ────────────────────────────────────────────────────

    def load_onnx_weights(self, onnx_path: str | Path | None = None):
        """Copy pre-trained weights from the installed swift_f0 ONNX model."""
        if onnx_path is None:
            onnx_path = _default_onnx_path()
        w = _onnx_weights(onnx_path)

        def _copy(param: nn.Parameter, key: str):
            arr = torch.from_numpy(w[key].copy())
            assert param.shape == arr.shape, \
                f"{key}: expected {param.shape}, got {arr.shape}"
            with torch.no_grad():
                param.copy_(arr)

        _copy(self.conv1.weight,           "onnx::Conv_152")
        _copy(self.conv1.bias,             "onnx::Conv_153")
        _copy(self.conv2.weight,           "onnx::Conv_155")
        _copy(self.conv2.bias,             "onnx::Conv_156")
        _copy(self.conv3.weight,           "onnx::Conv_158")
        _copy(self.conv3.bias,             "onnx::Conv_159")
        _copy(self.conv4.weight,           "onnx::Conv_161")
        _copy(self.conv4.bias,             "onnx::Conv_162")
        _copy(self.conf_head.weight,       "onnx::Conv_164")
        _copy(self.conf_head.bias,         "onnx::Conv_165")
        _copy(self.freq_projection.weight, "freq_projection.weight")
        _copy(self.freq_projection.bias,   "freq_projection.bias")

        centers = torch.from_numpy(w["pitch_bin_centers"].copy())
        with torch.no_grad():
            self.pitch_bin_centers.copy_(centers)

        print(f"[SwiftF0] Loaded pre-trained weights from {onnx_path}")
        return self

    def freeze_conv(self):
        """Freeze conv stack; fine-tune only the pitch projection head."""
        for layer in (self.conv1, self.conv2, self.conv3, self.conv4, self.conf_head):
            for p in layer.parameters():
                p.requires_grad_(False)

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad_(True)

    # ── forward ──────────────────────────────────────────────────────────

    def _log_spec(self, audio: torch.Tensor) -> torch.Tensor:
        """audio (B, L) → log-magnitude slice (B, 132, T)."""
        # Replicate ONNX model's 384-sample symmetric padding
        audio = F.pad(audio, (PAD, PAD))
        stft = torch.stft(
            audio,
            n_fft=N_FFT,
            hop_length=HOP,
            win_length=N_FFT,
            window=self.window,
            center=False,
            return_complex=True,
        )                                          # (B, 513, T)
        mag = stft.abs()                           # (B, 513, T)
        mag = mag[:, FREQ_LOW:FREQ_HIGH, :]        # (B, 132, T)
        return torch.log(mag + 1e-9)               # (B, 132, T)

    def _softmax_to_pitch_conf(
        self, probs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        probs: (B, T, 200) softmax distribution
        Returns:
            pitch_hz    (B, T)
            confidence  (B, T)  — prob mass within ±CONF_HALF_WIN of argmax
        """
        BT = probs.shape[0] * probs.shape[1]
        p_flat = probs.reshape(BT, N_PITCH_BINS)            # (B*T, 200)

        argmax = p_flat.argmax(dim=-1)                      # (B*T,)
        dist   = (self.bin_indices.unsqueeze(0) -
                  argmax.float().unsqueeze(1)).abs()         # (B*T, 200)
        mask   = (dist <= CONF_HALF_WIN).float()            # (B*T, 200)

        masked     = p_flat * mask                          # (B*T, 200)
        conf_flat  = masked.sum(dim=-1)                     # (B*T,)
        norm_flat  = masked / (conf_flat.unsqueeze(1) + 1e-7)  # (B*T, 200)

        centers    = self.pitch_bin_centers.unsqueeze(0)    # (1, 200)
        pitch_flat = (norm_flat * centers).sum(dim=-1)      # (B*T,)

        B, T = probs.shape[:2]
        return pitch_flat.reshape(B, T), conf_flat.reshape(B, T)

    def forward(self, audio: torch.Tensor):
        """
        audio : (B, L) float32 at 16 kHz
        Returns:
            pitch_hz    (B, T)
            confidence  (B, T)    — differentiable prob mass near peak
            pitch_probs (B, 200, T) — for CE loss during training
        """
        log_spec = self._log_spec(audio)                    # (B, 132, T)

        x = log_spec.unsqueeze(1)                           # (B, 1, 132, T)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conf_head(x))                       # (B, 1, 132, T)
        x = x.squeeze(1)                                    # (B, 132, T)

        logits = self.freq_projection(x)                    # (B, 200, T)
        probs  = F.softmax(logits.permute(0, 2, 1), dim=-1) # (B, T, 200)

        pitch_hz, confidence = self._softmax_to_pitch_conf(probs)
        pitch_probs = probs.permute(0, 2, 1)                # (B, 200, T) for CE loss

        return pitch_hz, confidence, pitch_probs

    # ── export ───────────────────────────────────────────────────────────

    def export_onnx(self, out_path: str | Path, audio_len: int = 16_000 * 30):
        """Export fine-tuned model to ONNX (drop-in for swift_f0/model.onnx)."""
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
        print(f"[SwiftF0] Exported ONNX → {out_path}")
