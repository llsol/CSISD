"""
Segment transition matrices.

Two analyses:
  1. Intra-svara: transitions between consecutive segments WITHIN each svara.
     Also reports boundary distributions (which types start / end a svara).
  2. Full pitch curve: all consecutive segment transitions including
     cross-svara boundaries (last seg of svara_i → first seg of svara_{i+1}).

Run:
    python -m src.analysis.segment_transition_analysis
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import settings as S
from src.io.pitch_io import load_preprocessed_pitch, load_flat_regions, load_peaks
from src.io.annotation_io import load_annotations
from src.features.structural_embedding import (
    label_samples_sil_cp_sta,
    map_peaks_to_global_rows,
    restrict_peaks_to_slice,
    build_segments_for_one_svara,
)

TYPES    = ["CP", "SIL", "STAp", "STAt", "TRa", "TRd"]
TYPE_IDX = {t: i for i, t in enumerate(TYPES)}
N        = len(TYPES)

OUT_DIR  = Path("figures/structural_analysis/v1")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_svara_segments(recording_id: str) -> list[list[str]]:
    """
    Returns a list of svaras; each svara is a list of segment type strings
    in order, e.g. ['CP', 'STAp', 'STAt', 'TRd', 'CP'].
    """
    tonic_hz = S.SARASUDA_TONICS[recording_id]
    df_pitch  = load_preprocessed_pitch(recording_id, S.INTERIM_RECORDINGS, tonic_hz, convert_to_cents=True)
    df_flat   = load_flat_regions(recording_id=recording_id, root_dir=S.INTERIM_RECORDINGS)
    df_peaks  = load_peaks(recording_id=recording_id, root_dir=S.INTERIM_RECORDINGS)

    df_pitch = (
        df_pitch
        .join(df_flat.select(["time_rel_sec", "flat_region"]), on="time_rel_sec", how="left")
        .with_columns(pl.col("flat_region").fill_null(False))
        .with_row_index("row_idx")
    )

    ann_path  = S.DATA_CORPUS / recording_id / "raw" / f"{recording_id}_ann_svara.tsv"
    df_svaras = load_annotations(file_path=ann_path, annotation_type="svara", engine="polars")
    peak_row_map = map_peaks_to_global_rows(df_pitch, df_peaks)
    t_all = df_pitch["time_rel_sec"].to_numpy()

    svara_seqs: list[list[str]] = []
    for ann in df_svaras.iter_rows(named=True):
        t0, t1 = float(ann["start_time_sec"]), float(ann["end_time_sec"])
        mask     = (t_all >= t0) & (t_all <= t1)
        df_svara = df_pitch.filter(pl.Series(mask))
        if df_svara.is_empty():
            continue
        df_svara = label_samples_sil_cp_sta(df_svara)
        local_pm = restrict_peaks_to_slice(df_svara, peak_row_map)
        segs     = build_segments_for_one_svara(df_svara=df_svara, local_peak_map=local_pm)
        svara_seqs.append([s["type"] for s in segs])

    return svara_seqs


def load_all_svara_seqs() -> list[list[str]]:
    all_seqs = []
    for rid in S.SARASUDA_VARNAM:
        print(f"  {rid}...")
        all_seqs.extend(_load_svara_segments(rid))
    print(f"  Total svaras: {len(all_seqs)}")
    return all_seqs


# ---------------------------------------------------------------------------
# Transition counting
# ---------------------------------------------------------------------------

def intra_svara_transitions(
    svara_seqs: list[list[str]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      mat_intra  (N, N) int  — intra-svara transition counts
      starts     (N,)   int  — counts of each type as first segment
      ends       (N,)   int  — counts of each type as last segment
    """
    mat    = np.zeros((N, N), dtype=int)
    starts = np.zeros(N, dtype=int)
    ends   = np.zeros(N, dtype=int)

    for seq in svara_seqs:
        if not seq:
            continue
        starts[TYPE_IDX[seq[0]]] += 1
        ends[TYPE_IDX[seq[-1]]]  += 1
        for a, b in zip(seq[:-1], seq[1:]):
            mat[TYPE_IDX[a], TYPE_IDX[b]] += 1

    return mat, starts, ends


def full_curve_transitions(
    svara_seqs: list[list[str]],
) -> np.ndarray:
    """
    All consecutive transitions including across svara boundaries
    (last seg of svara_i → first seg of svara_{i+1}).
    Returns (N, N) int count matrix.
    """
    mat = np.zeros((N, N), dtype=int)

    # intra-svara
    for seq in svara_seqs:
        for a, b in zip(seq[:-1], seq[1:]):
            mat[TYPE_IDX[a], TYPE_IDX[b]] += 1

    # cross-svara boundaries
    for seq_a, seq_b in zip(svara_seqs[:-1], svara_seqs[1:]):
        if seq_a and seq_b:
            mat[TYPE_IDX[seq_a[-1]], TYPE_IDX[seq_b[0]]] += 1

    return mat


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _norm_row(mat: np.ndarray) -> np.ndarray:
    """Row-normalise to probabilities (skip zero rows)."""
    row_sums = mat.sum(axis=1, keepdims=True).astype(float)
    row_sums[row_sums == 0] = 1.0
    return mat / row_sums


def _heatmap(ax, mat_norm: np.ndarray, mat_counts: np.ndarray, title: str) -> None:
    im = ax.imshow(mat_norm, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    ax.set_xticks(range(N)); ax.set_xticklabels(TYPES, fontsize=9)
    ax.set_yticks(range(N)); ax.set_yticklabels(TYPES, fontsize=9)
    ax.set_xlabel("Next segment", fontsize=9)
    ax.set_ylabel("Current segment", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    for i in range(N):
        for j in range(N):
            c = mat_counts[i, j]
            if c == 0:
                continue
            color = "white" if mat_norm[i, j] > 0.6 else "black"
            ax.text(j, i, str(c), ha="center", va="center", fontsize=11, color=color)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="p(next | current)")


def plot_all(
    mat_intra:  np.ndarray,
    mat_full:   np.ndarray,
    starts:     np.ndarray,
    ends:       np.ndarray,
    out_path:   Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("Segment Transition Analysis — Sarasuda Varnam corpus", fontsize=12)

    # Top-left: intra-svara heatmap
    _heatmap(axes[0, 0], _norm_row(mat_intra), mat_intra, "Intra-svara transitions (row = p(next|current))")

    # Top-right: full pitch curve heatmap
    _heatmap(axes[0, 1], _norm_row(mat_full), mat_full, "Full pitch curve transitions (incl. cross-svara)")

    # Bottom-left: svara boundary — starts and ends
    ax = axes[1, 0]
    x  = np.arange(N)
    w  = 0.35
    ax.bar(x - w/2, starts, width=w, label="Svara start (first seg)", color="#4e9ac7")
    ax.bar(x + w/2, ends,   width=w, label="Svara end (last seg)",    color="#e07b39")
    ax.set_xticks(x); ax.set_xticklabels(TYPES, fontsize=9)
    ax.set_ylabel("Count"); ax.set_title("Boundary distributions", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    for xi, (s, e) in enumerate(zip(starts, ends)):
        if s: ax.text(xi - w/2, s + 1, str(s), ha="center", fontsize=11)
        if e: ax.text(xi + w/2, e + 1, str(e), ha="center", fontsize=11)

    # Bottom-right: cross-svara only (diff = full - intra)
    mat_cross = mat_full - mat_intra
    _heatmap(axes[1, 1], _norm_row(mat_cross), mat_cross, "Cross-svara boundary transitions only")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading svara segments...")
    svara_seqs = load_all_svara_seqs()

    mat_intra, starts, ends = intra_svara_transitions(svara_seqs)
    mat_full                = full_curve_transitions(svara_seqs)

    print("\nIntra-svara transition counts:")
    header = f"{'':6s}" + "".join(f"{t:7s}" for t in TYPES)
    print(header)
    for i, row_name in enumerate(TYPES):
        row = "".join(f"{mat_intra[i,j]:7d}" for j in range(N))
        print(f"{row_name:6s}{row}")

    print("\nSvara starts:", {TYPES[i]: int(starts[i]) for i in range(N) if starts[i]})
    print("Svara ends:  ", {TYPES[i]: int(ends[i])   for i in range(N) if ends[i]})

    total_intra = mat_intra.sum()
    total_cross = (mat_full - mat_intra).sum()
    print(f"\nTotal intra-svara transitions: {total_intra}")
    print(f"Total cross-svara transitions: {total_cross}")

    plot_all(mat_intra, mat_full, starts, ends,
             OUT_DIR / "30_segment_transitions.png")


if __name__ == "__main__":
    main()
