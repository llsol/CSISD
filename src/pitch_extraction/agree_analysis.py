"""
Agree-percentage matrices across extractor / source / threshold combinations.

For each pair (FTA-Net source) × (Swift-model source × threshold), computes
agree% = frames where |Δ_cents| < 50¢ / total_frames, for every recording in
RECORDING_SELECTION, and shows the result as annotated heatmaps.

Layout: 2 rows (swiftf0-scratch / swiftf0-ft) × 4 cols (src_a × src_b combos).
Each cell is a heatmap: rows = recordings (+mean), cols = thresholds.

Usage
-----
  python -m src.pitch_extraction.agree_analysis
  python -m src.pitch_extraction.agree_analysis --save
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import settings

# ── constants (mirrored from compare_pitch) ───────────────────────────────────

VOICED_MIN_HZ   = 50.0
TONIC_FALLBACK  = 200.0
AGREE_THR_CENTS = 50.0

THRESHOLD_LIST = [0.30, 0.50, 0.70, 0.80, 0.90, 0.95]

EXTRACTOR_CV_DIR = {
    "ftanet":         "ftanet",
    "swiftf0ft":      "swiftf0_finetune",
    "swiftf0scratch": "swiftf0_scratch",
}
# Suffix used in the actual .npy filename (set by each predict.py)
EXTRACTOR_FILE_SUFFIX = {
    "ftanet":         "ftanet",
    "swiftf0ft":      "swiftf0finetune",
    "swiftf0scratch": "swiftf0scratch",
}
SOURCE_SHORT = {"original": "orig", "as": "BS-RoF"}


# ── data helpers ──────────────────────────────────────────────────────────────

def _pitch_path(extractor: str, source: str, recording_id: str) -> Path:
    cv_dir = EXTRACTOR_CV_DIR[extractor]
    suffix = EXTRACTOR_FILE_SUFFIX[extractor]
    return (settings.INTERIM_PITCH_CV / cv_dir / source / recording_id
            / f"{recording_id}_{source}_{suffix}_raw.npy")


def _load(path: Path):
    data = np.load(path)
    if data.shape[1] >= 3:
        return data[:, 0], data[:, 1], data[:, 2]
    return data[:, 0], data[:, 1], None


def _hz_to_cents(f0: np.ndarray, tonic: float) -> np.ndarray:
    out = np.full_like(f0, np.nan, dtype=float)
    v = f0 > VOICED_MIN_HZ
    out[v] = 1200.0 * np.log2(f0[v] / tonic)
    return out


def _apply_thr(f0: np.ndarray, conf, thr: float) -> np.ndarray:
    if conf is None:
        return f0
    out = f0.copy()
    out[conf < thr] = 0.0
    return out


def _align(time_a, ca, time_b, cb):
    dt_a = np.median(np.diff(time_a)) if len(time_a) > 1 else 0.01
    dt_b = np.median(np.diff(time_b)) if len(time_b) > 1 else 0.016
    if abs(dt_a - dt_b) < 0.0005:
        n = min(len(time_a), len(time_b))
        return time_a[:n], ca[:n], cb[:n]
    if dt_a <= dt_b:
        return time_a, ca, interpolate.interp1d(
            time_b, cb, kind="linear", bounds_error=False, fill_value=np.nan)(time_a)
    return time_b, interpolate.interp1d(
        time_a, ca, kind="linear", bounds_error=False, fill_value=np.nan)(time_b), cb


# ── metric ────────────────────────────────────────────────────────────────────

def agree_pct(rec_id: str, ext_a: str, src_a: str,
              ext_b: str, src_b: str, thr: float) -> float | None:
    pa = _pitch_path(ext_a, src_a, rec_id)
    pb = _pitch_path(ext_b, src_b, rec_id)
    if not pa.exists() or not pb.exists():
        return None

    tonic = settings.RECORDING_SELECTION_TONICS.get(rec_id, TONIC_FALLBACK)
    ta, f0a, _    = _load(pa)
    tb, f0b, conf = _load(pb)

    ca = _hz_to_cents(f0a, tonic)
    cb = _hz_to_cents(_apply_thr(f0b, conf, thr), tonic)

    t, ca, cb = _align(ta, ca, tb, cb)
    total     = len(t)
    both      = np.isfinite(ca) & np.isfinite(cb)
    agree     = both & (np.abs(cb - ca) < AGREE_THR_CENTS)
    return agree.sum() / total * 100


def build_matrix(recordings: list[str], ext_a: str, src_a: str,
                 ext_b: str, src_b: str) -> np.ndarray:
    """Shape: (n_rec + 1) × n_thr  — last row is nanmean across recordings."""
    rows = []
    for rec in recordings:
        row = [agree_pct(rec, ext_a, src_a, ext_b, src_b, t) for t in THRESHOLD_LIST]
        rows.append([v if v is not None else np.nan for v in row])
    mat = np.array(rows)
    with np.errstate(all="ignore"):
        mean_row = np.nanmean(mat, axis=0, keepdims=True)
    return np.vstack([mat, mean_row])


def agree_pct_union(rec_id: str, ext_a: str, src_a: str,
                    ext_b: str, src_b: str, thr: float) -> float | None:
    """Agree% over the union of voiced frames (ignores frames where both are unvoiced)."""
    pa = _pitch_path(ext_a, src_a, rec_id)
    pb = _pitch_path(ext_b, src_b, rec_id)
    if not pa.exists() or not pb.exists():
        return None

    tonic = settings.RECORDING_SELECTION_TONICS.get(rec_id, TONIC_FALLBACK)
    ta, f0a, _    = _load(pa)
    tb, f0b, conf = _load(pb)

    ca = _hz_to_cents(f0a, tonic)
    cb = _hz_to_cents(_apply_thr(f0b, conf, thr), tonic)

    t, ca, cb   = _align(ta, ca, tb, cb)
    voiced_a    = np.isfinite(ca)
    voiced_b    = np.isfinite(cb)
    union       = voiced_a | voiced_b
    both        = voiced_a & voiced_b
    agree       = both & (np.abs(cb - ca) < AGREE_THR_CENTS)
    denom       = union.sum()
    return agree.sum() / denom * 100 if denom > 0 else None


def build_matrix_union(recordings: list[str], ext_a: str, src_a: str,
                       ext_b: str, src_b: str) -> np.ndarray:
    """Same shape as build_matrix but agree% over voiced union, not total frames."""
    rows = []
    for rec in recordings:
        row = [agree_pct_union(rec, ext_a, src_a, ext_b, src_b, t) for t in THRESHOLD_LIST]
        rows.append([v if v is not None else np.nan for v in row])
    mat = np.array(rows)
    with np.errstate(all="ignore"):
        mean_row = np.nanmean(mat, axis=0, keepdims=True)
    return np.vstack([mat, mean_row])


def voiced_pct(rec_id: str, extractor: str, source: str,
               thr: float | None) -> float | None:
    path = _pitch_path(extractor, source, rec_id)
    if not path.exists():
        return None
    _, f0, conf = _load(path)
    f0_thr = _apply_thr(f0, conf, thr) if thr is not None else f0
    return (f0_thr > VOICED_MIN_HZ).sum() / len(f0_thr) * 100


def build_voiced_by_source(recordings: list[str], extractor: str,
                           sources: list[str]) -> np.ndarray:
    """FTA-Net (no threshold): shape (n_rec+1) × n_src. Cols = sources."""
    rows = []
    for rec in recordings:
        rows.append([voiced_pct(rec, extractor, src, None) or np.nan
                     for src in sources])
    mat = np.array(rows)
    with np.errstate(all="ignore"):
        mean_row = np.nanmean(mat, axis=0, keepdims=True)
    return np.vstack([mat, mean_row])


def build_voiced_by_thr(recordings: list[str], extractor: str,
                        source: str) -> np.ndarray:
    """Swift models: shape (n_rec+1) × n_thr. Cols = THRESHOLD_LIST."""
    rows = []
    for rec in recordings:
        rows.append([voiced_pct(rec, extractor, source, t) or np.nan
                     for t in THRESHOLD_LIST])
    mat = np.array(rows)
    with np.errstate(all="ignore"):
        mean_row = np.nanmean(mat, axis=0, keepdims=True)
    return np.vstack([mat, mean_row])


# ── plotting ──────────────────────────────────────────────────────────────────

def _annotate_heatmap(ax, mat, row_labels, col_labels):
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=7, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.axhline(len(row_labels) - 1.5, color="white", lw=1.5)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isfinite(v):
                txt_color = "black" if 25 < v < 75 else "white"
                weight    = "bold" if i == mat.shape[0] - 1 else "normal"
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        fontsize=6, color=txt_color, fontweight=weight)


def plot_heatmap(ax, mat: np.ndarray, row_labels: list[str],
                 col_labels: list[str], title: str,
                 vmin: float = 0.0, vmax: float = 100.0) -> object:
    im = ax.imshow(mat, aspect="auto", vmin=vmin, vmax=vmax,
                   cmap="RdYlGn", interpolation="nearest")

    _annotate_heatmap(ax, mat, row_labels, col_labels)
    ax.set_title(title, fontsize=8, pad=4)
    return im


def plot_heatmap_voiced(ax, mat: np.ndarray, row_labels: list[str],
                        col_labels: list[str], title: str,
                        vmin: float = 0.0, vmax: float = 100.0) -> object:
    im = ax.imshow(mat, aspect="auto", vmin=vmin, vmax=vmax,
                   cmap="Blues", interpolation="nearest")
    _annotate_heatmap(ax, mat, row_labels, col_labels)
    ax.set_title(title, fontsize=8, pad=4)
    return im


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    recordings = settings.RECORDING_SELECTION
    rec_short  = [r.split("_")[2] for r in recordings]   # bdn, drn, psn, rkm, svd
    row_labels = rec_short + ["mean"]
    col_labels = [f"{t:.2f}" for t in THRESHOLD_LIST]

    # panels: (ext_b, src_a, src_b, subplot_title)
    scratch_panels = [
        ("swiftf0scratch", "original", "original", "A=FTA[orig]  B=scratch[orig]"),
        ("swiftf0scratch", "original", "as",        "A=FTA[orig]  B=scratch[BSRoF]"),
        ("swiftf0scratch", "as",       "original",  "A=FTA[BSRoF]  B=scratch[orig]"),
        ("swiftf0scratch", "as",       "as",         "A=FTA[BSRoF]  B=scratch[BSRoF]"),
    ]
    ft_panels = [
        ("swiftf0ft", "original", "original", "A=FTA[orig]  B=ft[orig]"),
        ("swiftf0ft", "original", "as",        "A=FTA[orig]  B=ft[BSRoF]"),
        ("swiftf0ft", "as",       "original",  "A=FTA[BSRoF]  B=ft[orig]"),
        ("swiftf0ft", "as",       "as",         "A=FTA[BSRoF]  B=ft[BSRoF]"),
    ]
    all_panels = scratch_panels + ft_panels   # 8 panels, 2 rows × 4 cols

    n_rows_heat = len(recordings) + 1   # recordings + mean
    cell_h = 0.42
    fig_h  = 1.5 + n_rows_heat * cell_h * 2 + 0.8   # two matrix rows

    fig, axes = plt.subplots(
        2, 4,
        figsize=(4.5 * 4, fig_h),
        squeeze=False,
    )
    fig.suptitle(
        f"Agree % (|Δ| < {AGREE_THR_CENTS:.0f}¢, over total frames)\n"
        f"FTA-Net (A) vs SwiftF0 variants (B)  —  threshold on B",
        fontsize=11, y=0.995,
    )

    row_titles = ["SwiftF0-scratch", "SwiftF0-finetune"]
    ims = []
    for row_idx, panel_row in enumerate([scratch_panels, ft_panels]):
        axes[row_idx][0].set_ylabel(
            row_titles[row_idx] + "\nRecording", fontsize=8
        )
        for col_idx, (ext_b, src_a, src_b, title) in enumerate(panel_row):
            ax = axes[row_idx][col_idx]
            print(f"  {title} ...", flush=True)
            mat = build_matrix(recordings, "ftanet", src_a, ext_b, src_b)
            im  = plot_heatmap(ax, mat, row_labels, col_labels, title)
            ims.append(im)
            ax.set_xlabel("Threshold (B)", fontsize=7)

    # ── Figure 2: Voiced % ────────────────────────────────────────────────────
    print("\nBuilding voiced% matrices ...", flush=True)

    # Row heights proportional to number of columns (to keep cell aspect consistent)
    fig2 = plt.figure(figsize=(14, fig_h))
    fig2.suptitle(
        f"Voiced %  (f0 > {VOICED_MIN_HZ:.0f} Hz, over total frames)\n"
        f"per extractor · source · threshold",
        fontsize=11, y=0.995,
    )
    gs2 = fig2.add_gridspec(
        3, 2,
        left=0.07, right=0.93, top=0.92, bottom=0.06,
        hspace=0.55, wspace=0.25,
    )

    # Row 0: FTA-Net (no threshold) — spans both cols, cols = sources
    ax_fta = fig2.add_subplot(gs2[0, :])
    mat_fta = build_voiced_by_source(recordings, "ftanet", ["original", "as"])
    im_v = plot_heatmap_voiced(ax_fta, mat_fta, row_labels,
                               ["original", "BS-RoFormer"],
                               "FTA-Net  (pre-thresholded, no conf column)")
    ax_fta.set_xlabel("Source", fontsize=7)

    # Rows 1–2: Swift models — cols = thresholds
    swift_rows = [
        ("swiftf0scratch", "SwiftF0-scratch"),
        ("swiftf0ft",      "SwiftF0-finetune"),
    ]
    ims_v = [im_v]
    for row_idx, (ext, label) in enumerate(swift_rows, start=1):
        for col_idx, src in enumerate(["original", "as"]):
            ax = fig2.add_subplot(gs2[row_idx, col_idx])
            src_label = "original" if src == "original" else "BS-RoFormer"
            print(f"  {label} [{src_label}] ...", flush=True)
            mat = build_voiced_by_thr(recordings, ext, src)
            im  = plot_heatmap_voiced(ax, mat, row_labels, col_labels,
                                      f"{label}  [{src_label}]")
            ims_v.append(im)
            ax.set_xlabel("Threshold", fontsize=7)
            if col_idx == 0:
                ax.set_ylabel("Recording", fontsize=7)

    # ── Figure 3: Agree % over voiced union ───────────────────────────────────
    print("\nBuilding agree/union matrices ...", flush=True)

    fig3, axes3 = plt.subplots(2, 4, figsize=(4.5 * 4, fig_h), squeeze=False)
    fig3.suptitle(
        f"Agree % over voiced union  (|Δ| < {AGREE_THR_CENTS:.0f}¢ / (voiced_A ∪ voiced_B))\n"
        f"FTA-Net (A) vs SwiftF0 variants (B)  —  threshold on B",
        fontsize=11, y=0.995,
    )

    ims3 = []
    for row_idx, panel_row in enumerate([scratch_panels, ft_panels]):
        axes3[row_idx][0].set_ylabel(row_titles[row_idx] + "\nRecording", fontsize=8)
        for col_idx, (ext_b, src_a, src_b, title) in enumerate(panel_row):
            ax = axes3[row_idx][col_idx]
            print(f"  {title} ...", flush=True)
            mat = build_matrix_union(recordings, "ftanet", src_a, ext_b, src_b)
            im  = plot_heatmap(ax, mat, row_labels, col_labels, title)
            ims3.append(im)
            ax.set_xlabel("Threshold (B)", fontsize=7)


    out_dir = settings.FIGURES_DIR / "agree_analysis"
    if args.save:
        out_dir.mkdir(parents=True, exist_ok=True)
        p1 = out_dir / "agree_matrix.png"
        p2 = out_dir / "voiced_matrix.png"
        p3 = out_dir / "agree_union_matrix.png"
        fig.savefig(p1, dpi=150, bbox_inches="tight")
        fig2.savefig(p2, dpi=150, bbox_inches="tight")
        fig3.savefig(p3, dpi=150, bbox_inches="tight")
        print(f"→ {p1}")
        print(f"→ {p2}")
        print(f"→ {p3}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
