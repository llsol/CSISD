"""
Statistical analysis of segment shape parameters.

STA / TR
--------
Fits the 3-parameter tanh+osc model to each GT segment, then studies
(k, s, A) in relation to k-neighbours.  Neighbour distance is counted over
STA/TR only: CP and SIL segments are invisible to the counter.

  k=1 → each segment paired with the immediately next STA/TR
  k=2 → STA1 paired with STA3 (skipping STA2), whether or not there is a CP/TR in between

CP
--
Concavity = quadratic coefficient of a parabola fit to the flat region:
  a > 0  concave up   (pitch dips in the middle)
  a < 0  concave down / arch (pitch peaks in the middle)

Analysed in relation to:
  delta_next  = next_STA.end_cents  − CP.last_cents   (next STA directly after CP/TR)
  delta_prev  = CP.first_cents − prev_STA_or_CP.last_cents  (prev STA or CP, skipping TR)

Outputs
-------
  data/interim/analysis/segment_shapes.parquet
  data/interim/analysis/sta_tr_pairs_k{k}.parquet
  data/interim/analysis/cp_neighbours.parquet
  figures/analysis/shape_sta_tr_k{k}.png
  figures/analysis/shape_cp.png

Usage
-----
    python -m src.analysis.segment_shape_analysis
    python -m src.analysis.segment_shape_analysis --k 1 --k 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.optimize import curve_fit

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
    assign_segment_cents,
)

INTERIM_DIR = S.INTERIM_ANALYSIS
FIGURES_DIR = S.FIGURES_DIR / "analysis"
PITCH_COL   = "f0_savgol_p3_w13_cents"

# ── parametric model ─────────────────────────────────────────────────────────

def _h(t: np.ndarray, k: float, s: float, A: float) -> np.ndarray:
    return np.tanh(k * (t - s)) + A * np.sin(2 * np.pi * (t - 0.5))

def _curve_model(t: np.ndarray, k: float, s: float, A: float) -> np.ndarray:
    h0 = _h(np.array([0.0]), k, s, A)[0]
    h1 = _h(np.array([1.0]), k, s, A)[0]
    denom = h1 - h0
    return (_h(t, k, s, A) - h0) / denom if abs(denom) > 1e-9 else np.zeros_like(t)

def _fit_parametric(t_norm: np.ndarray, p_norm: np.ndarray) -> tuple[float, float, float, float]:
    """Returns (k, s, A, r2). Returns (nan,nan,nan,nan) if fit fails."""
    best_popt, best_rmse = None, np.inf
    for k0 in [1.0, 3.0, 8.0]:
        for s0 in [0.2, 0.5, 0.8]:
            for A0 in [0.0, -0.3]:
                try:
                    popt, _ = curve_fit(
                        _curve_model, t_norm, p_norm,
                        p0=[k0, s0, A0],
                        bounds=([0.3, 0.01, -0.4], [10.0, 0.99, 0.2]),
                        maxfev=2000,
                    )
                    rmse = float(np.sqrt(np.mean((_curve_model(t_norm, *popt) - p_norm) ** 2)))
                    if rmse < best_rmse:
                        best_rmse, best_popt = rmse, popt
                except Exception:
                    pass
    if best_popt is None:
        return np.nan, np.nan, np.nan, np.nan
    k, s, A = best_popt
    pred  = _curve_model(t_norm, k, s, A)
    ss_res = float(np.sum((p_norm - pred) ** 2))
    ss_tot = float(np.sum((p_norm - p_norm.mean()) ** 2))
    r2    = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0
    return float(k), float(s), float(A), float(r2)


def _concavity(cents: np.ndarray) -> float:
    """Quadratic coefficient of parabola fit to cents (normalized time)."""
    if len(cents) < 3:
        return 0.0
    t = np.linspace(0.0, 1.0, len(cents))
    return float(np.polyfit(t, cents, 2)[0])


# ── corpus extraction ─────────────────────────────────────────────────────────

def _extract_one_recording(
    recording_id: str,
    tonic_hz: float,
    min_samples_sta_tr: int = 6,
) -> list[dict]:
    """
    Returns one dict per non-SIL segment across all svaras in this recording.
    Each dict contains full shape info (params for STA/TR, concavity for CP)
    plus position metadata needed for neighbour construction.
    """
    interim = S.INTERIM_RECORDINGS
    corpus  = S.DATA_CORPUS

    df_pitch = load_preprocessed_pitch(
        recording_id=recording_id, root_dir=interim,
        tonic_hz=tonic_hz, convert_to_cents=True,
    )
    df_flat  = load_flat_regions(recording_id=recording_id, root_dir=interim)
    df_peaks = load_peaks(recording_id=recording_id, root_dir=interim)

    df_pitch = (
        df_pitch
        .join(df_flat.select(["time_rel_sec", "flat_region"]), on="time_rel_sec", how="left")
        .with_columns(pl.col("flat_region").fill_null(False))
        .with_row_index("row_idx")
    )

    ann_path  = corpus / recording_id / "raw" / f"{recording_id}_ann_svara.tsv"
    df_svaras = load_annotations(ann_path, annotation_type="svara", engine="polars")
    peak_row_map = map_peaks_to_global_rows(df_pitch, df_peaks)
    t_all = df_pitch["time_rel_sec"].to_numpy()

    records: list[dict] = []

    for svara_ann_idx, ann in enumerate(df_svaras.iter_rows(named=True)):
        t_start = float(ann["start_time_sec"])
        t_end   = float(ann["end_time_sec"])
        if t_end < t_start:
            t_start, t_end = t_end, t_start

        mask     = (t_all >= t_start) & (t_all <= t_end)
        df_svara = df_pitch.filter(pl.Series(mask))
        if df_svara.is_empty():
            continue

        df_svara   = label_samples_sil_cp_sta(df_svara)
        local_pkm  = restrict_peaks_to_slice(df_svara, peak_row_map)
        segments   = build_segments_for_one_svara(df_svara, local_pkm)
        segments   = assign_segment_cents(segments, df_svara, local_pkm)

        pitch_arr  = df_svara[PITCH_COL].to_numpy()
        times_arr  = df_svara["time_rel_sec"].to_numpy()

        for seg_idx, seg in enumerate(segments):
            if seg["type"] == "SIL":
                continue

            s, e = seg["start"], seg["end"]
            raw  = pitch_arr[s:e]
            valid_mask = np.isfinite(raw)
            raw_valid  = raw[valid_mask]
            if len(raw_valid) == 0:
                continue

            t_seg  = times_arr[s:e][valid_mask]
            dur    = float(t_seg[-1] - t_seg[0]) if len(t_seg) > 1 else 0.0

            base = {
                "recording_id":  recording_id,
                "svara_label":   ann["svara_label"],
                "svara_ann_idx": svara_ann_idx,
                "seg_idx":       seg_idx,
                "seg_type":      seg["type"],
                "dur_sec":       dur,
                "start_cents":   float(raw_valid[0]),
                "end_cents":     float(raw_valid[-1]),
                "mean_cents":    float(np.nanmean(raw_valid)),
                # shape params (filled below per type)
                "k_steep":   np.nan,
                "s_inflect": np.nan,
                "A_osc":     np.nan,
                "r2":        np.nan,
                "concavity": np.nan,
            }

            if seg["type"] == "CP":
                base["concavity"] = _concavity(raw_valid)

            elif seg["type"] in ("STAp", "STAt", "TRa", "TRd"):
                if len(raw_valid) < min_samples_sta_tr:
                    continue
                delta = raw_valid[-1] - raw_valid[0]
                if abs(delta) < 1e-6:
                    continue
                t_norm = np.linspace(0.0, 1.0, len(raw_valid))
                p_norm = (raw_valid - raw_valid[0]) / delta   # 0→1 for both STA and TR

                k, s, A, r2 = _fit_parametric(t_norm, p_norm)
                base.update({"k_steep": k, "s_inflect": s, "A_osc": A, "r2": r2})

            records.append(base)

    return records


def build_segment_dataset(
    recording_ids: list[str] = S.SARASUDA_VARNAM,
    tonic_map: dict[str, float] = S.SARASUDA_TONICS,
) -> pl.DataFrame:
    all_records: list[dict] = []
    for rid in recording_ids:
        print(f"  {rid}…")
        all_records.extend(_extract_one_recording(rid, tonic_map[rid]))
    df = pl.DataFrame(all_records)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    out = INTERIM_DIR / "segment_shapes.parquet"
    df.write_parquet(out)
    print(f"Saved {len(df)} segments → {out}")
    return df


# ── neighbour pair builders ───────────────────────────────────────────────────

def build_sta_tr_pairs(df: pl.DataFrame, k: int) -> pl.DataFrame:
    """
    Pair each STA/TR segment with its k-th STA/TR neighbour within the same svara.
    CP and SIL segments are invisible to the neighbour counter.
    """
    df_st  = df.filter(pl.col("seg_type").is_in(["STAp", "STAt", "TRa", "TRd"])).sort(
        ["recording_id", "svara_ann_idx", "seg_idx"]
    )
    rows_i, rows_j = [], []

    for (rid, sai), group in df_st.group_by(
        ["recording_id", "svara_ann_idx"], maintain_order=True
    ):
        g = group.to_dicts()
        for i in range(len(g) - k):
            rows_i.append(g[i])
            rows_j.append(g[i + k])

    if not rows_i:
        return pl.DataFrame()

    def _prefix(rows: list[dict], prefix: str) -> pl.DataFrame:
        keep = ["seg_type", "svara_label", "dur_sec",
                "start_cents", "end_cents", "k_steep", "s_inflect", "A_osc", "r2"]
        return pl.DataFrame(
            [{f"{prefix}{col}": r[col] for col in keep} for r in rows]
        )

    df_pairs = pl.concat([_prefix(rows_i, "i_"), _prefix(rows_j, "j_")], how="horizontal")
    df_pairs = df_pairs.with_columns(pl.lit(k).alias("k_neighbour"))

    out = INTERIM_DIR / f"sta_tr_pairs_k{k}.parquet"
    df_pairs.write_parquet(out)
    print(f"STA/TR pairs k={k}: {len(df_pairs)} pairs → {out}")
    return df_pairs


def build_cp_neighbours(df: pl.DataFrame) -> pl.DataFrame:
    """
    For each CP segment compute:
      delta_next  = next_STA.end_cents  − CP.last_cents   (next STA after CP, skipping TR)
      delta_prev  = CP.first_cents − prev_STA_or_CP.last_cents  (prev STA/CP, skipping TR)
    """
    df_sorted = df.sort(["recording_id", "svara_ann_idx", "seg_idx"])
    records: list[dict] = []

    for (rid, sai), group in df_sorted.group_by(
        ["recording_id", "svara_ann_idx"], maintain_order=True
    ):
        g = group.to_dicts()
        for i, seg in enumerate(g):
            if seg["seg_type"] != "CP":
                continue

            # delta_next: look forward, skip TR, find first STA
            delta_next = np.nan
            for j in range(i + 1, len(g)):
                if g[j]["seg_type"] in ("TRa", "TRd"):
                    continue
                if g[j]["seg_type"] in ("STAp", "STAt"):
                    delta_next = g[j]["end_cents"] - seg["end_cents"]
                break

            # delta_prev: look backward, skip TR, find first STA or CP
            delta_prev = np.nan
            for j in range(i - 1, -1, -1):
                if g[j]["seg_type"] in ("TRa", "TRd"):
                    continue
                if g[j]["seg_type"] in ("STAp", "STAt", "CP"):
                    delta_prev = seg["start_cents"] - g[j]["end_cents"]
                break

            records.append({
                "recording_id":  seg["recording_id"],
                "svara_label":   seg["svara_label"],
                "concavity":     seg["concavity"],
                "start_cents":   seg["start_cents"],
                "end_cents":     seg["end_cents"],
                "dur_sec":       seg["dur_sec"],
                "delta_next":    delta_next,
                "delta_prev":    delta_prev,
            })

    df_cp = pl.DataFrame(records)
    out   = INTERIM_DIR / "cp_neighbours.parquet"
    df_cp.write_parquet(out)
    print(f"CP neighbours: {len(df_cp)} rows → {out}")
    return df_cp


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_sta_tr(df: pl.DataFrame, df_pairs: pl.DataFrame, k: int) -> None:
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"STA/TR shape parameters  (k-neighbour = {k})", fontsize=11, fontweight="bold")

    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.35)

    # row 0: marginal distributions of k, s, A  (STA vs TR)
    for col_i, (param, label) in enumerate([
        ("k_steep",  "k (steepness)"),
        ("s_inflect","s (inflection)"),
        ("A_osc",    "A (oscillation)"),
    ]):
        ax = fig.add_subplot(gs[0, col_i])
        for stype, color in [("STAp", "#ff9800"), ("STAt", "#e65100"), ("TRa", "#2196f3"), ("TRd", "#0277bd")]:
            vals = df.filter(pl.col("seg_type") == stype)[param].drop_nulls().to_numpy()
            vals = vals[np.isfinite(vals)]
            ax.hist(vals, bins=30, alpha=0.6, color=color, label=stype, density=True)
        ax.set_title(label, fontsize=8)
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=6)

    # r2 histogram
    ax = fig.add_subplot(gs[0, 3])
    vals = df.filter(pl.col("r2").is_not_null())["r2"].drop_nulls().to_numpy()
    ax.hist(vals[np.isfinite(vals)], bins=30, color="#555", density=True)
    ax.set_title("R² fit quality", fontsize=8)
    ax.tick_params(labelsize=6)

    # row 1: k vs s scatter, colored by type; A vs s scatter
    if len(df_pairs) > 0:
        for col_i, (px, py, lx, ly) in enumerate([
            ("i_k_steep",  "i_s_inflect", "k (curr)",  "s (curr)"),
            ("i_k_steep",  "i_A_osc",     "k (curr)",  "A (curr)"),
            ("i_k_steep",  "j_k_steep",   "k (curr)",  f"k (k={k} next)"),
            ("i_s_inflect","j_s_inflect",  "s (curr)",  f"s (k={k} next)"),
        ]):
            ax = fig.add_subplot(gs[1, col_i])
            for stype, color in [("STAp", "#ff9800"), ("STAt", "#e65100"), ("TRa", "#2196f3"), ("TRd", "#0277bd")]:
                sub = df_pairs.filter(pl.col("i_seg_type") == stype)
                x = sub[px].drop_nulls().to_numpy()
                y = sub[py].drop_nulls().to_numpy()
                n = min(len(x), len(y))
                ax.scatter(x[:n], y[:n], alpha=0.3, s=8, color=color, label=stype)
            ax.set_xlabel(lx, fontsize=7)
            ax.set_ylabel(ly, fontsize=7)
            ax.tick_params(labelsize=6)
            if col_i == 0:
                ax.legend(fontsize=6)

    # row 2: A_osc comparisons
    if len(df_pairs) > 0:
        for col_i, (px, py, lx, ly) in enumerate([
            ("i_A_osc",    "j_A_osc",      "A (curr)",  f"A (k={k} next)"),
            ("i_seg_type", "i_A_osc",      "seg_type",  "A (curr)"),
        ]):
            ax = fig.add_subplot(gs[2, col_i])
            if px == "i_seg_type":
                for stype, color in [("STAp", "#ff9800"), ("STAt", "#e65100"), ("TRa", "#2196f3"), ("TRd", "#0277bd")]:
                    vals = df_pairs.filter(pl.col("i_seg_type") == stype)["i_A_osc"].to_numpy()
                    vals = vals[np.isfinite(vals)]
                    ax.hist(vals, bins=20, alpha=0.6, color=color, label=stype, density=True)
                ax.set_xlabel(lx, fontsize=7)
                ax.set_ylabel(ly, fontsize=7)
                ax.legend(fontsize=6)
            else:
                for stype, color in [("STAp", "#ff9800"), ("STAt", "#e65100"), ("TRa", "#2196f3"), ("TRd", "#0277bd")]:
                    sub = df_pairs.filter(pl.col("i_seg_type") == stype)
                    x = sub[px].to_numpy(); y = sub[py].to_numpy()
                    valid = np.isfinite(x) & np.isfinite(y)
                    ax.scatter(x[valid], y[valid], alpha=0.3, s=8, color=color, label=stype)
                ax.set_xlabel(lx, fontsize=7)
                ax.set_ylabel(ly, fontsize=7)
                if col_i == 0:
                    ax.legend(fontsize=6)
            ax.tick_params(labelsize=6)

        # transition type heatmap (i_type → j_type)
        ax = fig.add_subplot(gs[2, 2])
        types = ["STAp", "STAt", "TRa", "TRd"]
        n_t = len(types)
        mat = np.zeros((n_t, n_t))
        for ri, ti in enumerate(types):
            for ci, tj in enumerate(types):
                mat[ri, ci] = len(df_pairs.filter(
                    (pl.col("i_seg_type") == ti) & (pl.col("j_seg_type") == tj)
                ))
        ax.imshow(mat, cmap="Blues")
        ax.set_xticks(range(n_t)); ax.set_yticks(range(n_t))
        ax.set_xticklabels(types, fontsize=6); ax.set_yticklabels(types, fontsize=6)
        ax.set_xlabel(f"j (k={k})", fontsize=7); ax.set_ylabel("i (curr)", fontsize=7)
        ax.set_title("Transition counts", fontsize=8)
        for ri in range(n_t):
            for ci in range(n_t):
                ax.text(ci, ri, f"{int(mat[ri,ci])}", ha="center", va="center", fontsize=7)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / f"shape_sta_tr_k{k}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"→ {out}")
    plt.close(fig)


def plot_cp(df_cp: pl.DataFrame) -> None:
    df = df_cp.filter(pl.col("concavity").is_not_null())
    conc  = df["concavity"].to_numpy()
    dnext = df["delta_next"].to_numpy()
    dprev = df["delta_prev"].to_numpy()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("CP concavity analysis", fontsize=10, fontweight="bold")

    # distribution of concavity
    axes[0].hist(conc[np.isfinite(conc)], bins=40, color="#4caf50", density=True, alpha=0.8)
    axes[0].axvline(0, color="k", lw=0.8, ls="--")
    axes[0].set_xlabel("Concavity (quadratic coeff a)", fontsize=8)
    axes[0].set_ylabel("Density", fontsize=8)
    axes[0].set_title("CP concavity distribution\na>0 concave up, a<0 arch", fontsize=8)

    # concavity vs delta_next (to next STA peak)
    valid = np.isfinite(conc) & np.isfinite(dnext)
    axes[1].scatter(conc[valid], dnext[valid], alpha=0.35, s=10, color="#ff9800")
    axes[1].axhline(0, color="k", lw=0.5, ls="--", alpha=0.5)
    axes[1].axvline(0, color="k", lw=0.5, ls="--", alpha=0.5)
    axes[1].set_xlabel("CP concavity", fontsize=8)
    axes[1].set_ylabel("Δ next  (STA.end − CP.last) ¢", fontsize=8)
    axes[1].set_title("Concavity vs Δ to next STA", fontsize=8)
    if valid.sum() > 2:
        r = float(np.corrcoef(conc[valid], dnext[valid])[0, 1])
        axes[1].text(0.05, 0.95, f"r={r:.2f}", transform=axes[1].transAxes, fontsize=8, va="top")

    # concavity vs delta_prev (to prev STA or CP)
    valid = np.isfinite(conc) & np.isfinite(dprev)
    axes[2].scatter(conc[valid], dprev[valid], alpha=0.35, s=10, color="#2196f3")
    axes[2].axhline(0, color="k", lw=0.5, ls="--", alpha=0.5)
    axes[2].axvline(0, color="k", lw=0.5, ls="--", alpha=0.5)
    axes[2].set_xlabel("CP concavity", fontsize=8)
    axes[2].set_ylabel("Δ prev  (CP.first − prev.last) ¢", fontsize=8)
    axes[2].set_title("Concavity vs Δ from prev STA/CP", fontsize=8)
    if valid.sum() > 2:
        r = float(np.corrcoef(conc[valid], dprev[valid])[0, 1])
        axes[2].text(0.05, 0.95, f"r={r:.2f}", transform=axes[2].transAxes, fontsize=8, va="top")

    for ax in axes:
        ax.tick_params(labelsize=7)
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "shape_cp.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"→ {out}")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Segment shape parameter analysis.")
    parser.add_argument("--k", type=int, action="append", default=None,
                        help="Neighbour distance(s) for STA/TR pairing (default: 1). "
                             "Can be specified multiple times.")
    parser.add_argument("--rebuild", action="store_true",
                        help="Re-extract segment_shapes.parquet even if it already exists")
    args = parser.parse_args()

    k_values = args.k if args.k else [1]

    shapes_path = INTERIM_DIR / "segment_shapes.parquet"
    if shapes_path.exists() and not args.rebuild:
        print(f"Loading existing {shapes_path}")
        df = pl.read_parquet(shapes_path)
    else:
        print("Extracting segments from corpus…")
        df = build_segment_dataset()

    print(f"\n{len(df)} segments total")
    print(df.group_by("seg_type").agg(pl.len().alias("n")).sort("seg_type"))

    # STA/TR stats summary
    df_st = df.filter(pl.col("seg_type").is_in(["STAp", "STAt", "TRa", "TRd"]))
    print("\nSTAp/STAt/TRa/TRd parameter statistics:")
    print(
        df_st.group_by("seg_type")
        .agg(
            pl.col("k_steep").mean().alias("k_mean"),
            pl.col("k_steep").std().alias("k_std"),
            pl.col("s_inflect").mean().alias("s_mean"),
            pl.col("s_inflect").std().alias("s_std"),
            pl.col("A_osc").mean().alias("A_mean"),
            pl.col("A_osc").std().alias("A_std"),
            pl.col("r2").mean().alias("r2_mean"),
        )
        .sort("seg_type")
    )

    # CP stats summary
    df_cp_stats = df.filter(pl.col("seg_type") == "CP")
    print(f"\nCP concavity: mean={df_cp_stats['concavity'].mean():.2f}  "
          f"std={df_cp_stats['concavity'].std():.2f}  "
          f"n={len(df_cp_stats)}")

    # STA/TR neighbour pairs + plots
    for k in k_values:
        print(f"\nBuilding STA/TR pairs k={k}…")
        df_pairs = build_sta_tr_pairs(df, k)
        plot_sta_tr(df, df_pairs, k)

    # CP neighbour analysis + plot
    print("\nBuilding CP neighbour pairs…")
    df_cp_nb = build_cp_neighbours(df)
    plot_cp(df_cp_nb)


if __name__ == "__main__":
    main()
