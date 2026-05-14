"""
Full-piece structural segment plots.

Two segmentation strategies:
  svara-based  — pitch sliced per annotation, segments computed within each svara
  global       — pitch treated as one continuous stream, no svara division

Functions
---------
plot_piece_structural_segments(recording_id, tonic_hz, ...)
    Single panel: svara-based segmentation.

plot_piece_both_segmentations(recording_id, tonic_hz, ...)
    Two stacked panels: global (top) vs svara-based (bottom).

Usage
-----
    python -m src.utils.plot_piece_segments
    python -m src.utils.plot_piece_segments --recording srs_v1_bdn_sav
    python -m src.utils.plot_piece_segments --recording srs_v1_bdn_sav --both
    python -m src.utils.plot_piece_segments --all
    python -m src.utils.plot_piece_segments --recording srs_v1_svd_sav --out figures/piece.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import settings as S
from src.io.pitch_io import load_preprocessed_pitch, load_flat_regions, load_peaks
from src.io.annotation_io import load_annotations
from src.features.structural_embedding import (
    label_samples_sil_cp_sta,
    map_peaks_to_global_rows,
    restrict_peaks_to_slice,
    build_segments_for_one_svara,
    assign_segment_cents,
)

SEG_COLOR = {
    "CP":   ("#4caf50", 0.18),
    "STAp": ("#e91e8c", 0.20),
    "STAt": ("#c2185b", 0.20),
    "TRa":  ("#ff9800", 0.18),
    "TRd":  ("#e65100", 0.18),
    "SIL":  ("#9e9e9e", 0.12),
}
SVARA_COLORS = {
    "S": "#e6194b", "R": "#3cb44b", "G": "#4363d8",
    "M": "#f58231", "P": "#911eb4", "D": "#42d4f4", "N": "#f032e6",
}


# ── data loading ──────────────────────────────────────────────────────────────

def _load_data(
    recording_id: str,
    tonic_hz: float,
    corpus_root: Path,
    interim_root: Path,
    annotation_path: Path | None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, dict]:
    """Load pitch, peaks, annotations and global peak_row_map."""
    df_pitch = load_preprocessed_pitch(
        recording_id=recording_id,
        root_dir=interim_root,
        tonic_hz=tonic_hz,
        convert_to_cents=True,
    )
    df_flat  = load_flat_regions(recording_id=recording_id, root_dir=interim_root)
    df_peaks = load_peaks(recording_id=recording_id, root_dir=interim_root)

    df_pitch = (
        df_pitch
        .join(df_flat.select(["time_rel_sec", "flat_region"]), on="time_rel_sec", how="left")
        .with_columns(pl.col("flat_region").fill_null(False))
        .with_row_index("row_idx")
    )

    if annotation_path is None:
        annotation_path = (
            corpus_root / recording_id / "raw" / f"{recording_id}_ann_svara.tsv"
        )
    df_svaras    = load_annotations(annotation_path, annotation_type="svara", engine="polars")
    peak_row_map = map_peaks_to_global_rows(df_pitch, df_peaks)

    return df_pitch, df_peaks, df_svaras, peak_row_map


# ── segmentation strategies ───────────────────────────────────────────────────

def _segs_to_abs_times(segs: list[dict], times: np.ndarray) -> list[dict]:
    """Convert segment start/end local indices to absolute time values."""
    N  = len(times)
    dt = float(times[1] - times[0]) if N > 1 else 0.01
    out = []
    for seg in segs:
        s, e = seg["start"], seg["end"]
        t0 = float(times[s])
        t1 = float(times[e]) if e < N else float(times[N - 1] + dt)
        out.append({**seg, "t_start": t0, "t_end": t1})
    return out


def compute_svara_segments(
    df_pitch: pl.DataFrame,
    df_svaras: pl.DataFrame,
    peak_row_map: dict,
) -> list[dict]:
    """Segment pitch within each annotated svara independently."""
    t_all    = df_pitch["time_rel_sec"].to_numpy()
    all_segs = []

    for ann in df_svaras.iter_rows(named=True):
        t0 = float(ann["start_time_sec"])
        t1 = float(ann["end_time_sec"])
        if t1 < t0:
            t0, t1 = t1, t0
        sv_label = ann["svara_label"]

        mask  = (t_all >= t0) & (t_all <= t1)
        df_sv = df_pitch.filter(pl.Series(mask))
        if df_sv.is_empty():
            continue

        df_sv       = label_samples_sil_cp_sta(df_sv)
        local_peaks = restrict_peaks_to_slice(df_sv, peak_row_map)
        segs        = build_segments_for_one_svara(df_sv, local_peaks)
        segs        = assign_segment_cents(segs, df_sv, local_peaks)

        for seg in _segs_to_abs_times(segs, df_sv["time_rel_sec"].to_numpy()):
            all_segs.append({**seg, "svara_label": sv_label})

    return all_segs


def compute_global_segments(
    df_pitch: pl.DataFrame,
    peak_row_map: dict,
) -> list[dict]:
    """Segment the full pitch curve as a single continuous stream."""
    df_labelled = label_samples_sil_cp_sta(df_pitch)
    # For the full recording, local indices == row positions in df_pitch,
    # so restrict_peaks_to_slice maps global→local trivially.
    local_peaks = restrict_peaks_to_slice(df_labelled, peak_row_map)
    segs        = build_segments_for_one_svara(df_labelled, local_peaks)
    segs        = assign_segment_cents(segs, df_labelled, local_peaks)

    return _segs_to_abs_times(segs, df_pitch["time_rel_sec"].to_numpy())


# ── axis drawing helper ───────────────────────────────────────────────────────

def _draw_ax(
    ax: plt.Axes,
    t_all: np.ndarray,
    y_all: np.ndarray,
    segments: list[dict],
    df_svaras: pl.DataFrame,
    df_peaks: pl.DataFrame,
    title: str,
    show_peaks: bool = True,
) -> None:
    y_finite = y_all[np.isfinite(y_all)]
    y_lo     = float(np.nanmin(y_finite)) if len(y_finite) else -200.0
    y_top    = float(np.nanmax(y_finite)) if len(y_finite) else 1200.0
    y_hi     = y_top + 120   # room for svara labels inside the axes

    ax.set_ylim(y_lo - 60, y_hi)
    y_svara_label = y_hi - 15   # inside the axes, just below the top
    y_tr_label    = y_lo - 30   # inside the axes, just above the bottom

    # pitch contour
    ax.plot(t_all, y_all, lw=0.8, color="#333333", zorder=3)

    # shaded segments + type labels
    y_text_lo = y_lo - 50   # clamp bounds — keeps labels inside axes
    y_text_hi = y_hi - 20
    for seg in segments:
        color, alpha = SEG_COLOR.get(seg["type"], ("#000000", 0.1))
        ax.axvspan(seg["t_start"], seg["t_end"], color=color, alpha=alpha, lw=0)
        xm = (seg["t_start"] + seg["t_end"]) / 2
        if seg["type"] in ("CP", "STAp", "STAt") and np.isfinite(seg["cents"]):
            y_pos = float(np.clip(seg["cents"], y_text_lo, y_text_hi))
            label = seg["type"][:4]
            ax.text(xm, y_pos, label,
                    ha="center", va="bottom", fontsize=5, color="black", zorder=5,
                    clip_on=True)
        elif seg["type"] in ("TRa", "TRd"):
            ax.text(xm, y_tr_label, seg["type"],
                    ha="center", va="bottom", fontsize=5, color="black", zorder=5,
                    clip_on=True)

    # peaks
    if show_peaks and not df_peaks.is_empty():
        pt = df_peaks["time_savgol"].to_numpy()
        pv = df_peaks["value_savgol_cents"].to_numpy()
        pk = df_peaks["extremum_kind"].to_numpy()
        ax.scatter(pt[pk == "max"], pv[pk == "max"], marker="^", s=14,
                   color="#e53935", zorder=4, linewidths=0)
        ax.scatter(pt[pk == "min"], pv[pk == "min"], marker="v", s=14,
                   color="#1e88e5", zorder=4, linewidths=0)

    # svara boundaries + labels (inside axes, near top)
    prev_t1 = None
    for ann in df_svaras.iter_rows(named=True):
        t0  = float(ann["start_time_sec"])
        t1  = float(ann["end_time_sec"])
        if t1 < t0:
            t0, t1 = t1, t0
        sv  = ann["svara_label"]
        col = SVARA_COLORS.get(sv, "#000000")

        if prev_t1 is None or abs(t0 - prev_t1) > 0.01:
            ax.axvline(t0, color=col, lw=0.5, alpha=0.6, zorder=2)
        ax.axvline(t1, color=col, lw=0.5, alpha=0.6, zorder=2)
        ax.text((t0 + t1) / 2, y_svara_label, sv,
                ha="center", va="top", fontsize=7,
                color=col, fontweight="bold", zorder=6)
        prev_t1 = t1

    # reference lines
    ax.axhline(0,    color="steelblue", lw=0.5, ls="--", alpha=0.4)
    ax.axhline(1200, color="steelblue", lw=0.5, ls=":",  alpha=0.3)

    # legend
    patches = [
        mpatches.Patch(color=c, alpha=a + 0.3, label=t)
        for t, (c, a) in SEG_COLOR.items()
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=7, ncol=4)

    ax.set_ylabel("Cents rel. tonic", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.set_xlim(t_all[0], t_all[-1])


# ── public plot functions ─────────────────────────────────────────────────────

def plot_piece_structural_segments(
    recording_id: str,
    tonic_hz: float,
    corpus_root: Path | str = "data/corpus",
    interim_root: Path | str | None = None,
    annotation_path: Path | str | None = None,
    figsize: tuple[int, int] = (36, 5),
    show_peaks: bool = True,
) -> plt.Figure:
    """Single panel: svara-based segmentation."""
    corpus_root  = Path(corpus_root)
    import settings as _S
    interim_root = _S.INTERIM_RECORDINGS if interim_root is None else Path(interim_root)

    df_pitch, df_peaks, df_svaras, peak_row_map = _load_data(
        recording_id, tonic_hz, corpus_root, interim_root,
        Path(annotation_path) if annotation_path else None,
    )
    segments = compute_svara_segments(df_pitch, df_svaras, peak_row_map)
    t_all    = df_pitch["time_rel_sec"].to_numpy()
    y_all    = df_pitch["f0_savgol_p3_w13_cents"].to_numpy()

    fig, ax = plt.subplots(figsize=figsize)
    _draw_ax(ax, t_all, y_all, segments, df_svaras, df_peaks,
             title=f"{recording_id} — svara-based segmentation",
             show_peaks=show_peaks)
    ax.set_xlabel("Time (s)", fontsize=8)
    fig.tight_layout()
    return fig


def plot_piece_both_segmentations(
    recording_id: str,
    tonic_hz: float,
    corpus_root: Path | str = "data/corpus",
    interim_root: Path | str | None = None,
    annotation_path: Path | str | None = None,
    figsize: tuple[int, int] = (36, 9),
    show_peaks: bool = True,
) -> plt.Figure:
    """Two stacked panels: global segmentation (top) vs svara-based (bottom)."""
    corpus_root  = Path(corpus_root)
    import settings as _S
    interim_root = _S.INTERIM_RECORDINGS if interim_root is None else Path(interim_root)

    df_pitch, df_peaks, df_svaras, peak_row_map = _load_data(
        recording_id, tonic_hz, corpus_root, interim_root,
        Path(annotation_path) if annotation_path else None,
    )
    t_all = df_pitch["time_rel_sec"].to_numpy()
    y_all = df_pitch["f0_savgol_p3_w13_cents"].to_numpy()

    segs_global = compute_global_segments(df_pitch, peak_row_map)
    segs_svara  = compute_svara_segments(df_pitch, df_svaras, peak_row_map)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=figsize, sharex=True,
        gridspec_kw={"hspace": 0.08},
    )
    _draw_ax(ax_top, t_all, y_all, segs_global, df_svaras, df_peaks,
             title=f"{recording_id} — global segmentation (no svara boundary)",
             show_peaks=show_peaks)
    _draw_ax(ax_bot, t_all, y_all, segs_svara, df_svaras, df_peaks,
             title=f"{recording_id} — svara-based segmentation",
             show_peaks=show_peaks)
    ax_bot.set_xlabel("Time (s)", fontsize=8)
    fig.tight_layout()
    return fig


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot full-piece structural segments."
    )
    parser.add_argument("--recording", default=None,
                        help="Recording ID (default: first in SARASUDA_VARNAM)")
    parser.add_argument("--both",  action="store_true",
                        help="Show both segmentation strategies (two panels)")
    parser.add_argument("--all",   action="store_true",
                        help="Plot all recordings → PNG files")
    parser.add_argument("--out",   default=None,
                        help="Save to this PNG path instead of showing")
    parser.add_argument("--show",  action="store_true",
                        help="Open interactive window after saving (for --all or --out)")
    args = parser.parse_args()

    if args.all:
        if not args.show:
            matplotlib.use("Agg")
        out_dir = S.FIGURES_DIR / "structural_segments"
        out_dir.mkdir(parents=True, exist_ok=True)
        for rec_id in S.SARASUDA_VARNAM:
            tonic = S.SARASUDA_TONICS[rec_id]
            print(f"Plotting {rec_id} …")
            fig = plot_piece_both_segmentations(rec_id, tonic)
            out = out_dir / f"{rec_id}_both.png"
            fig.savefig(out, dpi=120, bbox_inches="tight")
            print(f"  → {out}")
            plt.show() if args.show else plt.close(fig)
    else:
        rec_id = args.recording or S.SARASUDA_VARNAM[0]
        tonic  = S.SARASUDA_TONICS[rec_id]
        print(f"Plotting {rec_id} …")
        plot_fn = plot_piece_both_segmentations if args.both else plot_piece_structural_segments
        fig = plot_fn(rec_id, tonic)
        if args.out:
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=120, bbox_inches="tight")
            print(f"Saved → {out}")
            if args.show:
                plt.show()
        else:
            plt.show()
