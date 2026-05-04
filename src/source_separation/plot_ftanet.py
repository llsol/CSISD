"""
Plot a single FTA-Net pitch extraction and save as PNG.

Usage:
    python -m src.source_separation.plot_ftanet
    python -m src.source_separation.plot_ftanet srs_v1_svd_sav
    python -m src.source_separation.plot_ftanet srs_v1_svd_sav --unet

Reads:
    data/interim/{id}/pitch_raw/{id}_ftanet_raw.npy        (default)
    data/interim/{id}/pitch_raw/{id}_unet_ftanet_raw.npy   (--unet)

Saves:
    data/interim/{id}/pitch_raw/{id}_ftanet_plot.png
    data/interim/{id}/pitch_raw/{id}_unet_ftanet_plot.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import settings

VOICED_MIN_HZ = 50.0
TONIC_FALLBACK = 200.0


def hz_to_cents(f0_hz: np.ndarray, tonic_hz: float) -> np.ndarray:
    out = np.full_like(f0_hz, np.nan, dtype=float)
    voiced = f0_hz > VOICED_MIN_HZ
    out[voiced] = 1200.0 * np.log2(f0_hz[voiced] / tonic_hz)
    return out


SOURCE_LABELS = {
    "original":   ("Original",       "steelblue"),
    "unet":       ("U-Net voice",    "tomato"),
    "as":         ("BS-RoFormer",    "seagreen"),
}


def plot_pitch(recording_id: str, source: str):
    pitch_dir = settings.DATA_INTERIM / recording_id / "pitch_raw"
    suffix    = "ftanet" if source == "original" else f"{source}_ftanet"
    npy_path  = pitch_dir / f"{recording_id}_{suffix}_raw.npy"

    if not npy_path.exists():
        raise FileNotFoundError(f"Pitch file not found: {npy_path}")

    data   = np.load(npy_path)
    time   = data[:, 0]
    f0_hz  = data[:, 1]
    tonic  = settings.SARASUDA_TONICS.get(recording_id, TONIC_FALLBACK)
    cents  = hz_to_cents(f0_hz, tonic)

    voiced_pct         = np.isfinite(cents).mean() * 100
    label, color       = SOURCE_LABELS.get(source, (source, "slategray"))

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(time, cents, lw=0.5, color=color, alpha=0.85)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cents re tonic")
    ax.set_title(f"FTA-Net pitch — {recording_id} [{label}]  "
                 f"(tonic={tonic:.1f} Hz, voiced={voiced_pct:.1f}%)")
    ax.grid(axis="y", lw=0.4, alpha=0.4)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("recording_id", nargs="?", default=settings.CURRENT_PIECE)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--unet", action="store_true", help="Plot U-Net separated voice pitch")
    source.add_argument("--as",   dest="as_model", action="store_true", help="Plot BS-RoFormer separated voice pitch")
    args = parser.parse_args()

    if args.unet:
        src = "unet"
    elif args.as_model:
        src = "as"
    else:
        src = "original"
    plot_pitch(args.recording_id, src)


if __name__ == "__main__":
    main()
