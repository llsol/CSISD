"""
Training loop for the Spectrogram-Channels U-Net (Oh et al. 2018).

Usage:
    python -m src.source_separation.train_unet

Data sources:
    settings.SSD_ROOT/**/Audio-Multitracks-Clean/vocals.wav
    settings.SSD_ROOT/**/Audio-Multitracks-Clean/tanpura.wav

Saved checkpoints:
    data/interim/source_separation/checkpoint_best.pt   (model state dict only)
    data/interim/source_separation/checkpoint_last.pt   (full state for resuming)

Sample log:
    data/interim/source_separation/sample_log.parquet
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import librosa
import polars as pl
import torch
from torch.utils.data import DataLoader, random_split

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import settings

from src.source_separation.unet import UNetSmall
from src.source_separation.dataset import TampuraSeparationDataset


# -----------------------------------------------------------------------
# Hyperparameters
# -----------------------------------------------------------------------
SR           = 22050
N_FFT        = 1024
HOP_LENGTH   = 256
PATCH_FRAMES = 128
ALPHA_RANGE  = (0.3, 1.0)

BATCH_SIZE       = 8
EPOCHS           = 40         # 20 @ LR + 20 @ LR/10 (paper schedule)
LR               = 1e-3
VAL_SPLIT        = 0.15
PATCHES_PER_PAIR = 200

BASE  = 32
ALPHA = 0.5   # equal weight: both sources are peak-normalised before mixing

CLEAN_SUBDIR     = "Audio-Multitracks-Clean"
CONCERT          = None    # None = all concerts on SSD
USE_PITCH_SHIFT  = True    # set False to bypass pitch augmentation entirely
OUT_DIR          = settings.DATA_INTERIM / "source_separation"
# -----------------------------------------------------------------------


def detect_tonic(tanpura_path: Path, sr: int = 22050) -> float:
    """Estimate tanpura tonic using pyin on a short segment."""
    audio, _ = librosa.load(tanpura_path, sr=sr, mono=True, duration=15, offset=5)
    f0, voiced, _ = librosa.pyin(audio, fmin=50, fmax=400, sr=sr)
    f0_voiced = f0[voiced & np.isfinite(f0)]
    return float(np.median(f0_voiced)) if len(f0_voiced) else 98.0


def shift_range_for_tonic(tonic_hz: float) -> tuple[float, float]:
    """
    Return (shift_min, shift_max) in semitones so that all tonic groups
    together cover C2–C3 in three equal 4-semitone bands:

        C2–E2   (65.4–82.4 Hz)  →  lowest group   → shift [0,   +4]
        E2–G#2  (82.4–103.8 Hz) →  middle group   → shift [-3,  +1]
        G#2–C3  (103.8–130.8 Hz)→  highest group  → shift [-1,  +3]
    """
    if tonic_hz < 82.4:     # C2 group
        return 0.0, 4.0
    elif tonic_hz < 103.8:  # G2 group (E2–G#2 band)
        return -3.0, 1.0
    else:                   # A2 group (G#2–C3 band)
        return -1.0, 3.0


def no_shift(_: float) -> tuple[float, float]:
    """Shift range used when USE_PITCH_SHIFT=False."""
    return 0.0, 0.0


def find_pairs(ssd_root: Path, concert: str | None = None) -> list[tuple[Path, Path, float, float]]:
    """
    Return matched (vocals.wav, tanpura.wav, shift_min, shift_max) tuples
    from the same piece folder. Trash directories are excluded.
    Tonic is auto-detected from the tanpura file to assign the shift range.
    """
    root = ssd_root / concert if concert else ssd_root
    pairs = []
    shift_fn = shift_range_for_tonic if USE_PITCH_SHIFT else no_shift

    for voice_file in sorted(root.glob(f"**/{CLEAN_SUBDIR}/vocals.wav")):
        if ".Trash" in voice_file.parts:
            continue
        tanpura_file = voice_file.parent / "tanpura.wav"
        if not tanpura_file.exists():
            continue
        tonic          = detect_tonic(tanpura_file, sr=SR)
        shift_min, shift_max = shift_fn(tonic)
        pairs.append((voice_file, tanpura_file, shift_min, shift_max))
        print(f"  {voice_file.parts[-3]:<35}  tonic={tonic:.1f} Hz  "
              f"shift=[{shift_min:+.1f}, {shift_max:+.1f}] st")

    return pairs


def weighted_l1(pred: torch.Tensor, voice: torch.Tensor,
                tanpura: torch.Tensor, alpha: float) -> torch.Tensor:
    """Weighted L1 loss: alpha * L1(voice) + (1-alpha) * L1(tanpura)."""
    l_voice   = (pred[:, 0:1] - voice).abs().mean()
    l_tanpura = (pred[:, 1:2] - tanpura).abs().mean()
    return alpha * l_voice + (1 - alpha) * l_tanpura


def train_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    for mag_mix, mag_voice, mag_tanpura, _ in loader:
        mag_mix     = mag_mix.to(device)
        mag_voice   = mag_voice.to(device)
        mag_tanpura = mag_tanpura.to(device)

        pred = model(mag_mix)
        loss = weighted_l1(pred, mag_voice, mag_tanpura, ALPHA)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def val_epoch(model, loader, device):
    model.eval()
    total = 0.0
    for mag_mix, mag_voice, mag_tanpura, _ in loader:
        mag_mix     = mag_mix.to(device)
        mag_voice   = mag_voice.to(device)
        mag_tanpura = mag_tanpura.to(device)
        pred        = model(mag_mix)
        total      += weighted_l1(pred, mag_voice, mag_tanpura, ALPHA).item()
    return total / len(loader)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not settings.SSD_ROOT.exists():
        raise FileNotFoundError(f"SSD not found: {settings.SSD_ROOT}")

    print(f"Detecting tonics and building pairs "
          f"({'with' if USE_PITCH_SHIFT else 'without'} pitch shift)...")
    pairs = find_pairs(settings.SSD_ROOT, concert=CONCERT)
    print(f"Matched pairs: {len(pairs)}")
    if not pairs:
        raise FileNotFoundError("No matched voice/tanpura pairs found on SSD.")

    dataset = TampuraSeparationDataset(
        pairs            = pairs,
        sr               = SR,
        n_fft            = N_FFT,
        hop_length       = HOP_LENGTH,
        patch_frames     = PATCH_FRAMES,
        alpha_range      = ALPHA_RANGE,
        patches_per_pair = PATCHES_PER_PAIR,
    )
    n_val   = max(1, int(len(dataset) * VAL_SPLIT))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Dataset: {len(dataset)} total  |  train: {n_train}  |  val: {n_val}")

    model = UNetSmall(in_channels=1, out_channels=2, base=BASE).to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[EPOCHS // 2], gamma=0.1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    best_val    = float("inf")
    best_path   = OUT_DIR / "checkpoint_best.pt"
    last_path   = OUT_DIR / "checkpoint_last.pt"
    start_epoch = 1
    all_logs: list[dict] = []

    # Resume from last checkpoint if available
    if last_path.exists():
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        best_val    = ckpt["best_val"]
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {ckpt['epoch']}  (best_val={best_val:.4f})")

    for epoch in range(start_epoch, EPOCHS + 1):
        dataset.current_epoch = epoch
        dataset.reset_log()

        t0         = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss   = val_epoch(model, val_loader, device)
        lr_now     = optimizer.param_groups[0]["lr"]
        scheduler.step()
        elapsed    = time.time() - t0

        all_logs.extend(dataset._sample_log)

        flag = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)
            flag = "  <- best"

        # Full checkpoint for resuming (saved every epoch)
        torch.save({
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val":             best_val,
        }, last_path)

        print(f"Epoch {epoch:3d}/{EPOCHS}  train={train_loss:.4f}  val={val_loss:.4f}  "
              f"lr={lr_now:.2e}  {elapsed:.0f}s{flag}")

    # if last_path.exists():
    #     last_path.unlink()

    # Save sample log
    if all_logs:
        pl.DataFrame(all_logs).write_parquet(OUT_DIR / "sample_log.parquet")
        print(f"Sample log saved: {len(all_logs):,} entries")

    print(f"\nBest model saved to: {best_path}  (val={best_val:.4f})")


if __name__ == "__main__":
    main()
