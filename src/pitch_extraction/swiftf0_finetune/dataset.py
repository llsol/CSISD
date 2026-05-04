"""
SCMS dataset loader for SwiftF0 fine-tuning.

SCMS format (Plaja-Roglans et al., 2023):
    audio/   *.wav  — 44.1 kHz, ~30 s, Carnatic vocal mix with resynthesised voice
    annotations/melody/*.csv  — "timestamp_sec, f0_Hz" (0 Hz = unvoiced), 29 ms hop

This loader:
1. Resamples audio to 16 kHz (SwiftF0 native SR).
2. Interpolates pitch annotations from the 29 ms SCMS grid to SwiftF0's 16 ms grid.
3. Optionally returns a random crop of `crop_frames` frames, seeded by (epoch, idx)
   so crops are reproducible within an epoch but vary across epochs.
4. Returns (audio_16k, pitch_hz, voiced) tensors aligned frame-by-frame.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import librosa
except ImportError:
    raise ImportError("librosa required: pip install librosa")

# SwiftF0 frame grid
SR_MODEL  = 16_000
HOP       = 256
FRAME_SEC = HOP / SR_MODEL  # 0.016 s


def scms_official_split(scms_root: str | Path) -> tuple[list[str], list[str]]:
    """
    Return (train_stems, test_stems) using the SCMS official artist-based split.
    Avoids data leakage by ensuring all chunks of an artist land in the same partition.
    """
    scms_root = Path(scms_root)
    with open(scms_root / "metadata.json") as f:
        meta = json.load(f)
    with open(scms_root / "artists_to_track_mapping.json") as f:
        mapping = json.load(f)

    train_stems, test_stems = [], []
    for artist in meta.get("train", {}):
        train_stems.extend(mapping.get(artist, []))
    for artist in meta.get("test", {}):
        test_stems.extend(mapping.get(artist, []))
    return train_stems, test_stems


def _load_scms_annotation(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (times_sec, f0_hz) arrays from an SCMS melody CSV."""
    times, freqs = [], []
    with open(csv_path) as f:
        for row in csv.reader(f):
            t, hz = float(row[0].strip()), float(row[1].strip())
            times.append(t)
            freqs.append(hz)
    return np.array(times, dtype=np.float64), np.array(freqs, dtype=np.float32)


def _align_to_model_grid(
    ann_times: np.ndarray,
    ann_f0:    np.ndarray,
    audio_sec: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolate SCMS pitch annotations (29 ms grid) to SwiftF0's 16 ms grid.

    Returns:
        f0_aligned  (N,) float32 — Hz, 0.0 = unvoiced
        voiced      (N,) bool
    """
    n_frames    = int(audio_sec * SR_MODEL / HOP)
    model_times = np.arange(n_frames) * FRAME_SEC

    idx = np.searchsorted(ann_times, model_times, side="right") - 1
    idx = np.clip(idx, 0, len(ann_f0) - 1)
    f0_aligned = ann_f0[idx].astype(np.float32)

    voiced = f0_aligned > 0.0
    return f0_aligned, voiced


class SCMSDataset(Dataset):
    """
    One item = one SCMS chunk (≈30 s WAV + pitch annotation), optionally cropped.

    Returns dict:
        audio  (L,)  float32 — mono 16 kHz
        f0     (T,)  float32 — Hz per SwiftF0 frame (0.0 = unvoiced)
        voiced (T,)  bool
        stem   str   — base filename without extension

    Random cropping:
        Set crop_sec > 0 to return a random crop of that duration instead of the
        full chunk. Crops are seeded by (self.epoch, idx) — reproducible within
        an epoch, different across epochs. Set self.epoch before each epoch starts.
    """

    def __init__(
        self,
        scms_root:  str | Path,
        split:      Optional[list[str]] = None,
        max_sec:    float = 35.0,
        crop_sec:   float = 0.0,
    ):
        """
        Args:
            scms_root:  Path to SCMS root directory (contains audio/ + annotations/).
            split:      Optional list of stem names (without extension) to use.
                        If None, uses all available pairs.
            max_sec:    Drop chunks longer than this (avoids OOM on outliers).
            crop_sec:   If > 0, return a random crop of this duration per item.
                        Crops vary across epochs via self.epoch.
        """
        scms_root = Path(scms_root)
        audio_dir = scms_root / "audio"
        ann_dir   = scms_root / "annotations" / "melody"

        pairs = []
        for wav in sorted(audio_dir.glob("*.wav")):
            csv_p = ann_dir / (wav.stem + ".csv")
            if not csv_p.exists():
                continue
            if split is not None and wav.stem not in split:
                continue
            pairs.append((wav, csv_p))

        self.pairs      = pairs
        self.max_sec    = max_sec
        self.crop_frames = int(crop_sec * SR_MODEL / HOP) if crop_sec > 0 else 0
        self.epoch      = 0   # set by training loop before each epoch

        crop_info = f"  crop={crop_sec:.1f}s ({self.crop_frames} frames)" if self.crop_frames else ""
        print(f"[SCMSDataset] {len(pairs)} chunks{crop_info}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        wav_path, csv_path = self.pairs[idx]

        audio, _ = librosa.load(str(wav_path), sr=SR_MODEL, mono=True)
        audio_sec = len(audio) / SR_MODEL

        if audio_sec > self.max_sec:
            audio     = audio[:int(self.max_sec * SR_MODEL)]
            audio_sec = self.max_sec

        ann_times, ann_f0 = _load_scms_annotation(csv_path)
        f0, voiced = _align_to_model_grid(ann_times, ann_f0, audio_sec)

        if self.crop_frames > 0 and len(f0) > self.crop_frames:
            rng        = random.Random(self.epoch * 100_000 + idx)
            max_start  = len(f0) - self.crop_frames
            frame_start = rng.randint(0, max_start)
            sample_start = frame_start * HOP
            audio  = audio[sample_start : sample_start + self.crop_frames * HOP]
            f0     = f0    [frame_start  : frame_start  + self.crop_frames]
            voiced = voiced[frame_start  : frame_start  + self.crop_frames]

        return {
            "audio":  torch.from_numpy(audio),
            "f0":     torch.from_numpy(f0),
            "voiced": torch.from_numpy(voiced),
            "stem":   wav_path.stem,
        }


def collate_fn(batch: list[dict]) -> dict:
    """
    Pad audio and label sequences to the same length within a batch.
    Audio is zero-padded; labels beyond the true length are marked unvoiced.
    """
    max_audio  = max(b["audio"].shape[0] for b in batch)
    max_frames = max(b["f0"].shape[0]    for b in batch)

    audio_pad  = torch.zeros(len(batch), max_audio)
    f0_pad     = torch.zeros(len(batch), max_frames)
    voiced_pad = torch.zeros(len(batch), max_frames, dtype=torch.bool)

    for i, b in enumerate(batch):
        la, lf = b["audio"].shape[0], b["f0"].shape[0]
        audio_pad[i, :la]  = b["audio"]
        f0_pad[i, :lf]     = b["f0"]
        voiced_pad[i, :lf] = b["voiced"]

    return {
        "audio":  audio_pad,
        "f0":     f0_pad,
        "voiced": voiced_pad,
        "stems":  [b["stem"] for b in batch],
    }
