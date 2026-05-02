"""
Dataset for U-Net voice/tanpura separation training.

Each pair is a (vocals.wav, tanpura.wav) from the same piece folder,
ensuring the tanpura is always tuned to the vocalist's tonic.

Augmentation:
- Pitch shift via resample trick (exact, no phase-vocoder artifacts):
  pretend the audio was recorded at sr*2^(shift/12) Hz, then resample
  to sr. Applied identically to voice and tanpura to preserve their
  harmonic relationship. Can be disabled by passing shift_range=(0, 0)
  for all pairs or by setting USE_PITCH_SHIFT=False in train_unet.py.
- Random alpha (tanpura mixing scale).
- Random patch offset within the piece.

Reproducibility:
- All random parameters (shift, alpha, start) are drawn from a
  deterministic RNG seeded by (epoch, sample_idx). Set
  dataset.current_epoch before each epoch to get different but
  reproducible samples every epoch.
- dataset._sample_log accumulates one dict per sample; call
  dataset.reset_log() to clear it between epochs.
"""

from pathlib import Path

import numpy as np
import torch
import librosa
from torch.utils.data import Dataset


class TampuraSeparationDataset(Dataset):
    """
    Args:
        pairs            : list of (voice_path, tanpura_path, shift_min_st, shift_max_st)
        sr               : sample rate
        n_fft            : FFT size
        hop_length       : STFT hop
        patch_frames     : temporal patch length in frames
        alpha_range      : (min, max) tanpura mixing scale
        patches_per_pair : random patches sampled per pair per epoch
    """

    def __init__(
        self,
        pairs: list[tuple[Path, Path, float, float]],
        sr: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        patch_frames: int = 128,
        alpha_range: tuple[float, float] = (0.3, 1.0),
        patches_per_pair: int = 200,
    ):
        if not pairs:
            raise ValueError("pairs is empty")
        self.pairs            = [(Path(v), Path(t), float(s0), float(s1))
                                  for v, t, s0, s1 in pairs]
        self.sr               = sr
        self.n_fft            = n_fft
        self.hop_length       = hop_length
        self.patch_frames     = patch_frames
        self.alpha_range      = alpha_range
        self.patches_per_pair = patches_per_pair
        # Samples needed for exactly patch_frames STFT frames
        self._patch_samples   = (patch_frames - 1) * hop_length + n_fft
        self._pair_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.current_epoch    = 0
        self._sample_log: list[dict] = []

    def reset_log(self):
        self._sample_log = []

    # ------------------------------------------------------------------
    def _load_pair(self, pair_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Load, align, and peak-normalise a (voice, tanpura) pair. Cached."""
        if pair_idx in self._pair_cache:
            return self._pair_cache[pair_idx]

        voice_path, tanpura_path, _, _ = self.pairs[pair_idx]
        voice,   _ = librosa.load(voice_path,   sr=self.sr, mono=True)
        tanpura, _ = librosa.load(tanpura_path, sr=self.sr, mono=True)

        # Tile tanpura if shorter (drone is periodic, tiling is musically valid)
        if len(tanpura) < len(voice):
            reps    = int(np.ceil(len(voice) / len(tanpura)))
            tanpura = np.tile(tanpura, reps)
        min_len = min(len(voice), len(tanpura))
        voice   = voice[:min_len].astype(np.float32)
        tanpura = tanpura[:min_len].astype(np.float32)

        # Peak-normalise each source independently (deterministic, safe to cache)
        voice   /= (np.abs(voice).max()   + 1e-8)
        tanpura /= (np.abs(tanpura).max() + 1e-8)

        self._pair_cache[pair_idx] = (voice, tanpura)
        return voice, tanpura

    def _pitch_shift_segment(self, seg: np.ndarray, shift_st: float) -> np.ndarray:
        """
        Pitch shift via resample trick: pretend audio is at sr*factor Hz,
        resample to sr. Exact, no phase-vocoder artifacts. Tempo changes
        proportionally but the segment is trimmed back to _patch_samples.
        """
        if abs(shift_st) < 0.01:
            return seg[:self._patch_samples]
        factor  = 2.0 ** (shift_st / 12.0)
        orig_sr = round(self.sr * factor)
        shifted = librosa.resample(seg, orig_sr=orig_sr, target_sr=self.sr)
        if len(shifted) >= self._patch_samples:
            return shifted[:self._patch_samples]
        return np.pad(shifted, (0, self._patch_samples - len(shifted)))

    def _patch_stft(self, segment: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """STFT of a short segment → (mag, phase), shape (F, patch_frames)."""
        D     = librosa.stft(segment, n_fft=self.n_fft, hop_length=self.hop_length)
        mag   = np.abs(D)[:, :self.patch_frames]
        phase = np.angle(D)[:, :self.patch_frames]
        if mag.shape[1] < self.patch_frames:
            pad   = self.patch_frames - mag.shape[1]
            mag   = np.pad(mag,   ((0, 0), (0, pad)))
            phase = np.pad(phase, ((0, 0), (0, pad)))
        return mag, phase

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.pairs) * self.patches_per_pair

    def __getitem__(self, idx):
        pair_idx          = idx // self.patches_per_pair
        patch_within_pair = idx  % self.patches_per_pair
        _, _, shift_min, shift_max = self.pairs[pair_idx]

        # Deterministic RNG seeded by (epoch, idx): reproducible but varies each epoch
        rng   = np.random.default_rng(seed=self.current_epoch * 999_983 + idx)
        shift = float(rng.uniform(shift_min, shift_max))
        alpha = float(rng.uniform(*self.alpha_range))

        voice, tanpura = self._load_pair(pair_idx)

        # Extract slightly more samples to compensate for tempo change after resample
        factor    = 2.0 ** (shift / 12.0)
        n_input   = int(np.ceil(self._patch_samples * factor)) + 1
        max_start = max(0, len(voice) - n_input)
        start     = int(rng.integers(0, max_start + 1))

        v_seg = voice[start : start + n_input].copy()
        t_seg = tanpura[start : start + n_input].copy()

        # Apply identical pitch shift to both sources (preserves tonal relationship)
        v_seg = self._pitch_shift_segment(v_seg, shift)
        t_seg = self._pitch_shift_segment(t_seg, shift)

        m_seg = v_seg + alpha * t_seg

        mag_v, _       = self._patch_stft(v_seg)
        mag_t, _       = self._patch_stft(t_seg)
        mag_m, phase_m = self._patch_stft(m_seg)

        # Normalise to [0, 1] using mix scale (consistent across all three)
        norm   = mag_m.max() + 1e-8
        mag_m /= norm
        mag_v /= norm
        mag_t /= norm

        self._sample_log.append({
            "epoch":             self.current_epoch,
            "idx":               idx,
            "pair_idx":          pair_idx,
            "patch_within_pair": patch_within_pair,
            "voice_path":        str(self.pairs[pair_idx][0]),
            "sample_start":      start,
            "alpha":             alpha,
            "pitch_shift_st":    shift,
        })

        return (
            torch.tensor(mag_m[np.newaxis],   dtype=torch.float32),  # (1, F, T)
            torch.tensor(mag_v[np.newaxis],   dtype=torch.float32),  # (1, F, T)
            torch.tensor(mag_t[np.newaxis],   dtype=torch.float32),  # (1, F, T)
            torch.tensor(phase_m[np.newaxis], dtype=torch.float32),  # (1, F, T)
        )
