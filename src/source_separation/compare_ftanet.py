"""
Compare FTA-Net pitch extraction: original corpus audio vs U-Net separated voice.

Usage:
    python -m src.source_separation.compare_ftanet
    python -m src.source_separation.compare_ftanet srs_v1_rkm_sav

Reads:
    data/interim/{id}/pitch_raw/{id}_ftanet_raw.npy        (original)
    data/interim/{id}/pitch_raw/{id}_unet_ftanet_raw.npy   (u-net voice)

Shows:
    1. Both pitch curves overlaid (cents re tonic)
    2. Absolute difference in cents — with top-N most divergent regions highlighted
    3. Voiced/unvoiced agreement summary
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import settings

# ── parameters ────────────────────────────────────────────────────────────────
TOP_N_REGIONS   = 5      # how many high-divergence regions to annotate
REGION_SEC      = 10.0   # window size (s) used to rank divergence regions
VOICED_MIN_HZ   = 50.0   # f0 below this → treat as unvoiced
TONIC_FALLBACK  = 200.0  # Hz, used if recording_id not in SARASUDA_TONICS
# ──────────────────────────────────────────────────────────────────────────────


def hz_to_cents(f0_hz: np.ndarray, tonic_hz: float) -> np.ndarray:
    """Convert f0 array (Hz) to cents re tonic. Unvoiced (≤0) → NaN."""
    out = np.full_like(f0_hz, np.nan, dtype=float)
    voiced = f0_hz > VOICED_MIN_HZ
    out[voiced] = 1200.0 * np.log2(f0_hz[voiced] / tonic_hz)
    return out


def top_divergence_regions(
    time: np.ndarray,
    diff_cents: np.ndarray,
    n: int,
    window_sec: float,
) -> list[tuple[float, float, float]]:
    """
    Return top-N (start_sec, end_sec, mean_abs_diff) windows of width window_sec
    ranked by mean absolute difference (NaN-safe).
    """
    dt       = float(np.median(np.diff(time)))
    win_frames = max(1, int(window_sec / dt))
    scores   = []
    i = 0
    while i + win_frames <= len(diff_cents):
        chunk = diff_cents[i : i + win_frames]
        score = np.nanmean(np.abs(chunk))
        if np.isfinite(score):
            scores.append((time[i], time[min(i + win_frames - 1, len(time) - 1)], score))
        i += win_frames

    scores.sort(key=lambda x: -x[2])

    # Keep non-overlapping top-N
    selected = []
    for s, e, sc in scores:
        if all(e <= ps or s >= pe for ps, pe, _ in selected):
            selected.append((s, e, sc))
        if len(selected) == n:
            break
    return selected


def compare(recording_id: str):
    pitch_dir = settings.DATA_INTERIM / recording_id / "pitch_raw"
    orig_path = pitch_dir / f"{recording_id}_ftanet_raw.npy"
    unet_path = pitch_dir / f"{recording_id}_unet_ftanet_raw.npy"

    if not orig_path.exists():
        raise FileNotFoundError(f"Original pitch not found: {orig_path}")
    if not unet_path.exists():
        raise FileNotFoundError(f"U-Net pitch not found: {unet_path}\nRun ftanet_predict with --unet first.")

    orig = np.load(orig_path)   # (N, 2)
    unet = np.load(unet_path)   # (N, 2)

    # Align to shorter length
    n     = min(len(orig), len(unet))
    time  = orig[:n, 0]
    f0_orig = orig[:n, 1]
    f0_unet = unet[:n, 1]

    tonic = settings.SARASUDA_TONICS.get(recording_id, TONIC_FALLBACK)
    cents_orig = hz_to_cents(f0_orig, tonic)
    cents_unet = hz_to_cents(f0_unet, tonic)

    # Voiced masks
    voiced_orig = np.isfinite(cents_orig)
    voiced_unet = np.isfinite(cents_unet)
    both_voiced = voiced_orig & voiced_unet

    diff = np.where(both_voiced, cents_unet - cents_orig, np.nan)

    # Stats
    mae  = np.nanmean(np.abs(diff))
    bias = np.nanmean(diff)
    pct_both   = both_voiced.mean() * 100
    pct_orig_only = (voiced_orig & ~voiced_unet).mean() * 100
    pct_unet_only = (~voiced_orig & voiced_unet).mean() * 100

    print(f"\n{'─'*55}")
    print(f"Recording : {recording_id}")
    print(f"Tonic     : {tonic:.1f} Hz")
    print(f"Frames    : {n:,}  ({time[-1]:.1f} s)")
    print(f"Both voiced   : {pct_both:.1f}%")
    print(f"Orig only     : {pct_orig_only:.1f}%")
    print(f"U-Net only    : {pct_unet_only:.1f}%")
    print(f"MAE (cents)   : {mae:.1f}")
    print(f"Bias (cents)  : {bias:+.1f}  (U-Net − original)")
    print(f"{'─'*55}\n")

    regions = top_divergence_regions(time, diff, TOP_N_REGIONS, REGION_SEC)

    # ── plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    fig.suptitle(f"FTA-Net comparison — {recording_id}  (tonic={tonic:.1f} Hz)", fontsize=13)

    # 1. Pitch curves
    ax = axes[0]
    ax.plot(time, cents_orig, lw=0.6, color="steelblue", alpha=0.8, label="Original")
    ax.plot(time, cents_unet, lw=0.6, color="tomato",    alpha=0.8, label="U-Net voice")
    ax.set_ylabel("Cents re tonic")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Pitch curves")

    # 2. Absolute difference
    ax = axes[1]
    ax.fill_between(time, 0, np.abs(diff), color="orchid", alpha=0.7)
    ax.axhline(mae, color="purple", lw=1, ls="--", label=f"MAE={mae:.1f} ¢")
    ax.set_ylabel("|Δ| cents")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Absolute pitch difference (voiced frames only)")

    # 3. Signed difference
    ax = axes[2]
    ax.fill_between(time, 0, diff, where=diff > 0, color="tomato",   alpha=0.6, label="U-Net higher")
    ax.fill_between(time, 0, diff, where=diff < 0, color="steelblue", alpha=0.6, label="U-Net lower")
    ax.axhline(0,    color="black",  lw=0.8)
    ax.axhline(bias, color="purple", lw=1, ls="--", label=f"bias={bias:+.1f} ¢")
    ax.set_ylabel("Δ cents")
    ax.set_xlabel("Time (s)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Signed difference (U-Net − original)")

    # Highlight top-N regions across all axes
    colors = plt.cm.autumn(np.linspace(0.2, 0.8, len(regions)))
    for ax in axes:
        for (s, e, sc), col in zip(regions, colors):
            ax.axvspan(s, e, color=col, alpha=0.18)

    # Annotate on middle plot
    for i, ((s, e, sc), col) in enumerate(zip(regions, colors)):
        mid = (s + e) / 2
        axes[1].annotate(
            f"#{i+1}\n{sc:.0f}¢",
            xy=(mid, sc), xytext=(mid, sc + mae * 0.5),
            fontsize=7, ha="center", color="purple",
        )

    plt.tight_layout()
    plt.show()


def main():
    recording_id = sys.argv[1] if len(sys.argv) > 1 else settings.CURRENT_PIECE
    compare(recording_id)


if __name__ == "__main__":
    main()
