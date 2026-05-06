"""
Plot pitch curves for a SCMS stem: ground truth vs FTA-Net (original) vs FTA-Net (BS-RoFormer).

Useful for inspecting whether high VFA values are true false alarms or GT annotation gaps.

Run:
    python -m src.source_separation.scms.plot Dorakuna_1
    python -m src.source_separation.scms.plot Dorakuna_1 --start 5 --end 15
    python -m src.source_separation.scms.plot Dorakuna_1 --sources original as

Reads:
    data/datasets/scms/annotations/melody/{stem}.csv        ← ground truth (29 ms hop)
    data/interim/scms_pitch/{stem}_original_ftanet_raw.npy  ← FTA-Net / original audio
    data/interim/scms_pitch/{stem}_as_ftanet_raw.npy        ← FTA-Net / BS-RoFormer separated
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import interpolate

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import settings
from src.pitch_extraction.swiftf0_finetune.dataset import scms_official_split

SCMS_ROOT  = settings.PROJECT_ROOT / "data" / "datasets" / "scms"
PITCH_ROOT = settings.DATA_INTERIM / "scms_pitch"

SOURCES = {
    "original": ("FTA-Net / original",      "#1565C0"),   # deep blue
    "as":       ("FTA-Net / BS-RoFormer",   "#BF360C"),   # deep orange-red
    "unet":     ("FTA-Net / U-Net",         "#2E7D32"),   # deep green
}
GT_COLOR   = "#BDBDBD"  # light gray
GT_LABEL   = "Ground truth (SCMS)"
GT_LW      = 3.0
PRED_LW    = 0.8
GT_ALPHA   = 0.9
PRED_ALPHA = 0.85
VOICED_HZ = 0.0


def load_annotation(path: Path) -> tuple[np.ndarray, np.ndarray]:
    times, freqs = [], []
    with open(path) as f:
        for row in csv.reader(f):
            times.append(float(row[0].strip()))
            freqs.append(float(row[1].strip()))
    return np.array(times), np.array(freqs)


def load_prediction(stem: str, source: str) -> tuple[np.ndarray, np.ndarray] | None:
    path = PITCH_ROOT / f"{stem}_{source}_ftanet_raw.npy"
    if not path.exists():
        return None
    arr = np.load(path)
    return arr[:, 0], arr[:, 1]


def _interp_to_grid(src_t, src_f, tgt_t):
    """Interpolate src_f to tgt_t grid; unvoiced (0 Hz) stays 0."""
    f = interpolate.interp1d(src_t, src_f, kind="linear",
                             bounds_error=False, fill_value=0.0)
    return f(tgt_t)


def plot_stem(stem: str, sources: list[str], t_start: float, t_end: float | None):
    ann_path = SCMS_ROOT / "annotations" / "melody" / f"{stem}.csv"
    if not ann_path.exists():
        print(f"[error] annotation not found: {ann_path}")
        return

    gt_t, gt_f = load_annotation(ann_path)

    # Apply time window
    mask = gt_t >= t_start
    if t_end is not None:
        mask &= gt_t <= t_end
    gt_t = gt_t[mask]
    gt_f = gt_f[mask]

    predictions: list[tuple[str, np.ndarray, np.ndarray, str, str]] = []
    for source in sources:
        pred = load_prediction(stem, source)
        if pred is None:
            print(f"[warn] no prediction for {stem}/{source} — run ftanet.py first")
            continue
        t, f = pred
        m = t >= t_start
        if t_end is not None:
            m &= t <= t_end
        label, color = SOURCES.get(source, (source, "gray"))
        predictions.append((source, t[m], f[m], label, color))

    if not predictions:
        print("[error] no predictions loaded")
        return

    # ── figure ────────────────────────────────────────────────────────────────
    n_panels = 2   # pitch curves + voicing agreement
    fig, axes = plt.subplots(n_panels, 1, figsize=(16, 8), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1]})
    title = f"Pitch curves — {stem}"
    if t_end is not None:
        title += f"  [{t_start:.1f}–{t_end:.1f} s]"
    fig.suptitle(title, fontsize=13)

    ax_pitch  = axes[0]
    ax_voice  = axes[1]

    # ── panel 1: pitch curves ─────────────────────────────────────────────────
    # Ground truth at the bottom
    gt_voiced = gt_f > 0
    ax_pitch.plot(gt_t[gt_voiced], gt_f[gt_voiced],
                  lw=GT_LW, alpha=GT_ALPHA, color=GT_COLOR,
                  zorder=2, label=GT_LABEL)

    # Predictions on top
    for source, t, f, label, color in predictions:
        voiced = f > 0
        ax_pitch.plot(t[voiced], f[voiced],
                      lw=PRED_LW, alpha=PRED_ALPHA, color=color,
                      zorder=4, label=label)

    ax_pitch.set_ylabel("f0 (Hz)")
    ax_pitch.legend(loc="upper right", fontsize=9)

    # ── panel 2: voicing agreement ────────────────────────────────────────────
    # For each source: shade where model=voiced but GT=unvoiced (potential VFA)
    # and where model=unvoiced but GT=voiced (missed)
    y_offsets = {src: i for i, (src, *_) in enumerate(predictions)}
    n_src = len(predictions)
    ax_voice.set_ylim(-0.5, n_src - 0.5)
    ax_voice.set_yticks(range(n_src))
    ax_voice.set_yticklabels(
        [label for _, _, _, label, _ in predictions], fontsize=8
    )

    for source, t, f, label, color in predictions:
        y = y_offsets[source]
        # Interpolate GT to this prediction's time grid
        gt_on_pred = _interp_to_grid(gt_t, gt_f, t)

        pred_voiced = f > 0
        gt_voiced_interp = gt_on_pred > 0

        # Both voiced — correct detection
        both = pred_voiced & gt_voiced_interp
        # VFA candidate: model=voiced, GT=unvoiced
        vfa  = pred_voiced & ~gt_voiced_interp
        # Miss: model=unvoiced, GT=voiced
        miss = ~pred_voiced & gt_voiced_interp

        for condition, c, alpha, zorder in [
            (miss, "#9E9E9E", 0.6, 2),   # gray — miss (bottom layer)
            (both, color,    0.85, 3),   # source color — correct
            (vfa,  "#E53935", 0.95, 4),  # bright red — VFA candidate (top)
        ]:
            if condition.any():
                ax_voice.scatter(
                    t[condition],
                    np.full(condition.sum(), y),
                    s=5, color=c, alpha=alpha,
                    linewidths=0, zorder=zorder,
                )

    # Legend for voicing panel
    patches = [
        mpatches.Patch(color="#1565C0", label="Both voiced (correct)"),
        mpatches.Patch(color="#E53935", label="Model voiced / GT unvoiced (VFA?)"),
        mpatches.Patch(color="#9E9E9E", label="Model unvoiced / GT voiced (miss)"),
    ]
    ax_voice.legend(handles=patches, loc="upper right", fontsize=8)
    ax_voice.set_xlabel("Time (s)")
    ax_voice.set_ylabel("Source")

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize pitch curves: GT vs FTA-Net across sources."
    )
    parser.add_argument("stem", help="SCMS stem (e.g. Dorakuna_1)")
    parser.add_argument("--sources", nargs="+", default=["original", "as"],
                        choices=list(SOURCES), help="Sources to plot (default: original as)")
    parser.add_argument("--start", type=float, default=0.0,
                        help="Start time in seconds (default: 0)")
    parser.add_argument("--end", type=float, default=None,
                        help="End time in seconds (default: full audio)")
    args = parser.parse_args()

    plot_stem(args.stem, args.sources, args.start, args.end)


if __name__ == "__main__":
    main()
