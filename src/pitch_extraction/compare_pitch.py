"""
Compare two pitch extractors (or sources) on a Carnatic recording.

Usage:
    # Same extractor, different sources:
    python -m src.pitch_extraction.compare_pitch srs_v1_svd_sav \\
        --extractor-a ftanet --extractor-b ftanet --source-a original --source-b as

    # Different extractors, same source:
    python -m src.pitch_extraction.compare_pitch srs_v1_svd_sav \\
        --extractor-a ftanet --extractor-b swiftf0ft --source as

    # Any combination:
    python -m src.pitch_extraction.compare_pitch srs_v1_svd_sav \\
        --extractor-a ftanet --source-a original --extractor-b swiftf0ft --source-b as

    # All RECORDING_SELECTION:
    python -m src.pitch_extraction.compare_pitch --all --source as

Reads:
    data/interim/cv_pitch_{extractor}/{source}/{id}/{id}_{source}_{extractor}_raw.npy

Plots:
    1. Pitch curves — color-coded by agreement / disagreement
    2. Absolute difference (both-voiced frames)
    3. Signed difference
    4. Histogram of exclusive voiced frames
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import interpolate

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import settings

# ── parameters ────────────────────────────────────────────────────────────────
TOP_N_REGIONS   = 5
REGION_SEC      = 10.0
VOICED_MIN_HZ   = 50.0
TONIC_FALLBACK  = 200.0
AGREE_THR_CENTS = 50.0   # agreement threshold: ±50¢ = one semitone

# Pitch curve color scheme (2-extractor case)
AGREE_COLOR  = "#909090"   # gray   — both agree within threshold
DISAG_A_COLOR = "#FF00CC"  # magenta — extractor A when diverging
DISAG_B_COLOR = "#00CCCC"  # cyan    — extractor B when diverging
ONLY_A_COLOR  = "#8B0000"  # dark crimson — only A voiced
ONLY_B_COLOR  = "#00008B"  # dark blue    — only B voiced

EXTRACTOR_STYLES = {
    "ftanet":        ("FTA-Net",         "steelblue"),
    "swiftf0":       ("SwiftF0",         "tomato"),
    "swiftf0ft":     ("SwiftF0-ft",      "gold"),
    "swiftf0scratch":("SwiftF0-scratch", "darkorange"),
}

# Subdirectory name under INTERIM_PITCH_CV/ for each extractor key.
EXTRACTOR_CV_DIR: dict[str, str] = {
    "ftanet":         "ftanet",
    "swiftf0":        "swiftf0",
    "swiftf0ft":      "swiftf0_finetune",
    "swiftf0scratch": "swiftf0_scratch",
}

SOURCE_LABELS = {
    "original": "original mix",
    "unet":     "U-Net separated",
    "as":       "BS-RoFormer separated",
}

MODEL_CONFIG = {
    "ftanet":  {"sample_rate": 8000,  "hop_size": 80,  "time_res_ms": 10},
    "swiftf0": {"sample_rate": 16000, "hop_size": 256, "time_res_ms": 16},
}
# ──────────────────────────────────────────────────────────────────────────────


def hz_to_cents(f0: np.ndarray, tonic: float) -> np.ndarray:
    out = np.full_like(f0, np.nan, dtype=float)
    voiced = f0 > VOICED_MIN_HZ
    out[voiced] = 1200.0 * np.log2(f0[voiced] / tonic)
    return out


def align_pitch_tracks(
    time_a: np.ndarray,
    pitch_a: np.ndarray,
    time_b: np.ndarray,
    pitch_b: np.ndarray,
    label_a: str = "A",
    label_b: str = "B",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Align two pitch tracks to the same time grid (the finer one)."""
    dt_a = np.median(np.diff(time_a)) if len(time_a) > 1 else 0.01
    dt_b = np.median(np.diff(time_b)) if len(time_b) > 1 else 0.016

    if abs(dt_a - dt_b) < 0.0005:
        n = min(len(time_a), len(time_b))
        return time_a[:n], pitch_a[:n], pitch_b[:n], "same resolution (truncated)"

    if dt_a <= dt_b:
        target_time = time_a
        pitch_b_aligned = interpolate.interp1d(
            time_b, pitch_b, kind="linear", bounds_error=False, fill_value=np.nan
        )(target_time)
        pitch_a_aligned = pitch_a
        info = f"aligned: {label_b} ({dt_b*1000:.1f}ms) → {label_a} ({dt_a*1000:.1f}ms) grid"
    else:
        target_time = time_b
        pitch_a_aligned = interpolate.interp1d(
            time_a, pitch_a, kind="linear", bounds_error=False, fill_value=np.nan
        )(target_time)
        pitch_b_aligned = pitch_b
        info = f"aligned: {label_a} ({dt_a*1000:.1f}ms) → {label_b} ({dt_b*1000:.1f}ms) grid"

    return target_time, pitch_a_aligned, pitch_b_aligned, info


def _top_divergence_regions(
    time: np.ndarray,
    diff: np.ndarray,
    n: int,
    window_sec: float,
) -> list[tuple[float, float, float]]:
    """Return the n non-overlapping windows with the highest mean |diff|."""
    dt  = float(np.median(np.diff(time))) if len(time) > 1 else 0.01
    win = max(1, int(window_sec / dt))
    scores = []
    i = 0
    while i + win <= len(diff):
        valid = diff[i : i + win]
        valid = valid[np.isfinite(valid)]
        if len(valid):
            scores.append((time[i], time[min(i + win - 1, len(time) - 1)],
                           np.mean(np.abs(valid))))
        i += win
    scores.sort(key=lambda x: -x[2])
    selected: list[tuple[float, float, float]] = []
    for s, e, sc in scores:
        if all(e <= ps or s >= pe for ps, pe, _ in selected):
            selected.append((s, e, sc))
        if len(selected) == n:
            break
    return selected


def _masked_line(ax, time, values, mask, color, lw=0.9, zorder=2, **kw):
    """Plot a line only where mask is True; NaN elsewhere creates natural breaks."""
    y = np.where(mask, values, np.nan)
    ax.plot(time, y, color=color, lw=lw, zorder=zorder, **kw)


def compare(
    recording_id: str,
    source_a: str = "unet",
    source_b: str | None = None,
    extractor_a: str = "ftanet",
    extractor_b: str = "swiftf0",
):
    if source_b is None:
        source_b = source_a

    cv_dir_a = EXTRACTOR_CV_DIR.get(extractor_a, extractor_a)
    cv_dir_b = EXTRACTOR_CV_DIR.get(extractor_b, extractor_b)
    path_a = settings.INTERIM_PITCH_CV / cv_dir_a / source_a / recording_id / f"{recording_id}_{source_a}_{extractor_a}_raw.npy"
    path_b = settings.INTERIM_PITCH_CV / cv_dir_b / source_b / recording_id / f"{recording_id}_{source_b}_{extractor_b}_raw.npy"

    label_a, color_a = EXTRACTOR_STYLES.get(extractor_a, (extractor_a, "steelblue"))
    label_b, color_b = EXTRACTOR_STYLES.get(extractor_b, (extractor_b, "tomato"))

    if source_a == source_b:
        source_label = SOURCE_LABELS.get(source_a, source_a)
    else:
        source_label = (f"{SOURCE_LABELS.get(source_a, source_a)} "
                        f"vs {SOURCE_LABELS.get(source_b, source_b)}")
        label_a = f"{label_a} / {SOURCE_LABELS.get(source_a, source_a)}"
        label_b = f"{label_b} / {SOURCE_LABELS.get(source_b, source_b)}"

    if not path_a.exists():
        raise FileNotFoundError(f"{label_a} pitch not found: {path_a}")
    if not path_b.exists():
        raise FileNotFoundError(f"{label_b} pitch not found: {path_b}")

    data_a = np.load(path_a)
    data_b = np.load(path_b)

    time_a, f0_a = data_a[:, 0], data_a[:, 1]
    time_b, f0_b = data_b[:, 0], data_b[:, 1]

    tonic       = settings.RECORDING_SELECTION_TONICS.get(recording_id, TONIC_FALLBACK)
    cents_a_raw = hz_to_cents(f0_a, tonic)
    cents_b_raw = hz_to_cents(f0_b, tonic)

    time, cents_a, cents_b, align_info = align_pitch_tracks(
        time_a, cents_a_raw, time_b, cents_b_raw, label_a, label_b
    )

    voiced_a    = np.isfinite(cents_a)
    voiced_b    = np.isfinite(cents_b)
    both_voiced = voiced_a & voiced_b
    diff        = np.where(both_voiced, cents_b - cents_a, np.nan)
    mae         = np.nanmean(np.abs(diff))
    bias        = np.nanmean(diff)

    total_frames = len(time)
    pct_both   = both_voiced.sum() / total_frames * 100
    pct_a_only = (voiced_a & ~voiced_b).sum() / total_frames * 100
    pct_b_only = (~voiced_a & voiced_b).sum() / total_frames * 100

    dt_a_real = np.median(np.diff(time_a)) * 1000
    dt_b_real = np.median(np.diff(time_b)) * 1000

    # Agreement / disagreement masks
    abs_diff   = np.where(both_voiced, np.abs(diff), np.nan)
    agree_mask = both_voiced & (abs_diff < AGREE_THR_CENTS)
    disag_mask = both_voiced & ~agree_mask
    only_a     = voiced_a & ~voiced_b
    only_b     = ~voiced_a & voiced_b

    print(f"\n{'─'*60}")
    print(f"Recording  : {recording_id}  [{source_label}]")
    print(f"Tonic      : {tonic:.1f} Hz")
    print(f"Duration   : {time[-1]:.1f} s")
    print(f"Resolution : {label_a}={dt_a_real:.1f}ms  {label_b}={dt_b_real:.1f}ms")
    print(f"Alignment  : {align_info}")
    print(f"Both voiced: {pct_both:.1f}%  |  "
          f"{label_a} only: {pct_a_only:.1f}%  |  {label_b} only: {pct_b_only:.1f}%")
    print(f"MAE        : {mae:.1f} ¢    Bias: {bias:+.1f} ¢  ({label_b} − {label_a})")
    print(f"{'─'*60}\n")

    regions      = _top_divergence_regions(time, diff, TOP_N_REGIONS, REGION_SEC)
    a_only_cents = cents_a[only_a]
    b_only_cents = cents_b[only_b]

    # ── figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 11))
    fig.suptitle(
        f"Pitch comparison — {recording_id}  "
        f"[{label_a} ({dt_a_real:.0f}ms) vs {label_b} ({dt_b_real:.0f}ms)]  "
        f"[{source_label}]  tonic={tonic:.1f} Hz",
        fontsize=13,
    )
    gs      = fig.add_gridspec(3, 2, width_ratios=[4, 1], hspace=0.35, wspace=0.08)
    ax0     = fig.add_subplot(gs[0, 0])
    ax1     = fig.add_subplot(gs[1, 0], sharex=ax0)
    ax2     = fig.add_subplot(gs[2, 0], sharex=ax0)
    ax_hist = fig.add_subplot(gs[:, 1])
    axes    = [ax0, ax1, ax2]

    # ── plot 1: pitch curves color-coded by agreement ─────────────────────────
    # agreement → gray line (both tracks overlap, draw once)
    _masked_line(ax0, time, cents_a, agree_mask, AGREE_COLOR,   lw=0.9, zorder=3)
    _masked_line(ax0, time, cents_b, agree_mask, AGREE_COLOR,   lw=0.9, zorder=3)
    # divergence → magenta (A) / cyan (B)
    _masked_line(ax0, time, cents_a, disag_mask, DISAG_A_COLOR, lw=0.8, zorder=2)
    _masked_line(ax0, time, cents_b, disag_mask, DISAG_B_COLOR, lw=0.8, zorder=2)
    # exclusive voiced → dark crimson (A) / dark blue (B)
    _masked_line(ax0, time, cents_a, only_a,     ONLY_A_COLOR,  lw=0.7, zorder=2)
    _masked_line(ax0, time, cents_b, only_b,     ONLY_B_COLOR,  lw=0.7, zorder=2)

    ax0.legend(handles=[
        mpatches.Patch(color=AGREE_COLOR,   label=f"agree <{AGREE_THR_CENTS:.0f}¢  (n={agree_mask.sum():,})"),
        mpatches.Patch(color=DISAG_A_COLOR, label=f"{label_a} diverges (magenta)"),
        mpatches.Patch(color=DISAG_B_COLOR, label=f"{label_b} diverges (cyan)"),
        mpatches.Patch(color=ONLY_A_COLOR,  label=f"only {label_a} voiced  ({pct_a_only:.1f}%)"),
        mpatches.Patch(color=ONLY_B_COLOR,  label=f"only {label_b} voiced  ({pct_b_only:.1f}%)"),
    ], loc="upper right", fontsize=7)
    ax0.set_ylabel("Cents re tonic")
    ax0.set_title(f"Pitch curves — {label_a} vs {label_b}  [{source_label}]")

    # ── plot 2: absolute difference ───────────────────────────────────────────
    ax1.fill_between(time, 0, np.abs(diff), where=np.isfinite(diff),
                     color="orchid", alpha=0.7)
    ax1.axhline(mae, color="purple", lw=1, ls="--", label=f"MAE={mae:.1f} ¢")
    ax1.set_ylabel("|Δ| cents")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_title(f"|Δ| pitch — both-voiced frames  ({label_a} vs {label_b})")

    # ── plot 3: signed difference — magenta (B>A) / cyan (B<A) ───────────────
    ax2.fill_between(time, 0, diff, where=np.isfinite(diff) & (diff > 0),
                     color=DISAG_A_COLOR, alpha=0.6, label=f"{label_b} higher (magenta)")
    ax2.fill_between(time, 0, diff, where=np.isfinite(diff) & (diff < 0),
                     color=DISAG_B_COLOR, alpha=0.6, label=f"{label_b} lower (cyan)")
    ax2.axhline(0,    color="black",  lw=0.8)
    ax2.axhline(bias, color="purple", lw=1, ls="--", label=f"bias={bias:+.1f} ¢")
    ax2.set_ylabel("Δ cents")
    ax2.set_xlabel("Time (s)")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.set_title(f"Signed difference ({label_b} − {label_a})  [{source_label}]")

    # ── top divergence region highlights ─────────────────────────────────────
    if regions:
        region_colors = plt.cm.autumn(np.linspace(0.2, 0.8, len(regions)))
        for ax in axes:
            for (s, e, _), col in zip(regions, region_colors):
                ax.axvspan(s, e, color=col, alpha=0.07)
        for i, ((s, e, sc), col) in enumerate(zip(regions, region_colors)):
            mid = (s + e) / 2
            axes[1].annotate(
                f"#{i+1}\n{sc:.0f}¢",
                xy=(mid, sc), xytext=(mid, sc + mae * 0.5),
                fontsize=7, ha="center", color="purple",
            )

    # ── plot 4: histogram of exclusive voiced frames ──────────────────────────
    bins = np.arange(-1200, 2401, 50)
    if len(a_only_cents):
        ax_hist.barh(bins[:-1], np.histogram(a_only_cents, bins=bins)[0],
                     height=48, color=ONLY_A_COLOR, alpha=0.8,
                     label=f"only {label_a}  ({pct_a_only:.1f}%)")
    if len(b_only_cents):
        ax_hist.barh(bins[:-1], -np.histogram(b_only_cents, bins=bins)[0],
                     height=48, color=ONLY_B_COLOR, alpha=0.8,
                     label=f"only {label_b}  ({pct_b_only:.1f}%)")
    ax_hist.axhline(0, color="black", lw=0.5)
    ax_hist.axvline(0, color="black", lw=0.8)
    ax_hist.set_xlabel(f"Frames  (← {label_b}   {label_a} →)")
    ax_hist.set_ylabel("Cents re tonic")
    ax_hist.legend(fontsize=8, loc="upper right")
    ax_hist.set_title(f"Exclusive voiced frames\n{label_a} vs {label_b}", fontsize=9)

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Compare two pitch extractors on the same audio source."
    )
    parser.add_argument("recordings", nargs="*",
                        help="Recording IDs (default: all RECORDING_SELECTION)")
    parser.add_argument("--all", action="store_true",
                        help="Process all recordings in settings.RECORDING_SELECTION")
    parser.add_argument("--source", default="unet",
                        help="Audio source for both A and B: original | unet | as")
    parser.add_argument("--source-a", default=None,
                        help="Audio source for extractor A (overrides --source)")
    parser.add_argument("--source-b", default=None,
                        help="Audio source for extractor B (overrides --source)")
    parser.add_argument("--extractor-a", default="ftanet",
                        help="First extractor (default: ftanet)")
    parser.add_argument("--extractor-b", default="swiftf0",
                        help="Second extractor (default: swiftf0)")
    args = parser.parse_args()

    sa = args.source_a or args.source
    sb = args.source_b or args.source

    recordings = (settings.RECORDING_SELECTION if (args.all or not args.recordings)
                  else args.recordings)

    for rec_id in recordings:
        compare(rec_id, source_a=sa, source_b=sb,
                extractor_a=args.extractor_a, extractor_b=args.extractor_b)


if __name__ == "__main__":
    main()
