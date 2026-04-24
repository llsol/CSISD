"""
Training loop del U-Net de separació de tampura.

Ús:
    python -m src.source_separation.train_unet

Estructura de dades esperada:
    data/source_separation/
        voice/      ← .wav de veu neta
        tampura/    ← .wav de tampura neta

El model entrenat es desa a:
    data/interim/source_separation/unet_best.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import settings

from src.source_separation.unet import UNetSmall
from src.source_separation.dataset import TampuraSeparationDataset


# -----------------------------------------------------------------------
# Hiperparàmetres
# -----------------------------------------------------------------------
SR           = 22050
N_FFT        = 1024
HOP_LENGTH   = 256
PATCH_FRAMES = 128
ALPHA_RANGE  = (0.3, 1.0)

BATCH_SIZE   = 8
EPOCHS       = 50
LR           = 1e-3
VAL_SPLIT    = 0.15

BASE         = 32   # filtres base del U-Net

VOICE_DIR    = settings.PROJECT_ROOT / "data" / "source_separation" / "voice"
TAMPURA_DIR  = settings.PROJECT_ROOT / "data" / "source_separation" / "tampura"
OUT_DIR      = settings.DATA_INTERIM / "source_separation"
# -----------------------------------------------------------------------


def l1_loss(pred_mag: torch.Tensor, target_mag: torch.Tensor) -> torch.Tensor:
    """L1,1: suma de valors absoluts de la diferència (eq. 1 del paper)."""
    return (pred_mag - target_mag).abs().mean()


def train_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    for mag_mix, mag_voice, _ in loader:
        mag_mix   = mag_mix.to(device)
        mag_voice = mag_voice.to(device)

        mask     = model(mag_mix)
        pred_mag = mask * mag_mix
        loss     = l1_loss(pred_mag, mag_voice)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def val_epoch(model, loader, device):
    model.eval()
    total = 0.0
    for mag_mix, mag_voice, _ in loader:
        mag_mix   = mag_mix.to(device)
        mag_voice = mag_voice.to(device)
        mask      = model(mag_mix)
        pred_mag  = mask * mag_mix
        total    += l1_loss(pred_mag, mag_voice).item()
    return total / len(loader)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset
    dataset = TampuraSeparationDataset(
        voice_dir    = VOICE_DIR,
        tampura_dir  = TAMPURA_DIR,
        sr           = SR,
        n_fft        = N_FFT,
        hop_length   = HOP_LENGTH,
        patch_frames = PATCH_FRAMES,
        alpha_range  = ALPHA_RANGE,
    )
    n_val   = max(1, int(len(dataset) * VAL_SPLIT))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    print(f"Train: {n_train} | Val: {n_val}")

    # Model
    model     = UNetSmall(in_channels=1, base=BASE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    best_val  = float("inf")
    best_path = OUT_DIR / "unet_best.pt"

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss   = val_epoch(model, val_loader, device)
        scheduler.step(val_loss)

        flag = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)
            flag = "  ← best"

        print(f"Epoch {epoch:3d}/{EPOCHS}  train={train_loss:.4f}  val={val_loss:.4f}{flag}")

    print(f"\nMillor model desat a: {best_path}  (val={best_val:.4f})")


if __name__ == "__main__":
    main()
