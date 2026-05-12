"""
Statistical analysis of CP/STA/TR segment shape parameters.

CP
--
  concavity  = quadratic coefficient a of parabola fit (cents/s²):
               a > 0  concave up (dips in middle)
               a < 0  arch (peaks in middle)
  Analysed by: svara, dur_sec bucket, performer.

STA / TR
--------
  Parametric model params: k_steep, s_inflect, A_osc
  Analysed by: svara, dur_sec bucket, performer.
  Produced separately and aggregated.

Inter-segment derivative analysis
----------------------------------
  Boundary slope (cents/s) computed analytically:
    STA/TR: from fitted (k, s, A) params
    CP:     from quadratic fit (concavity + start/end/dur)
    SIL:    slope = 0
  For each consecutive pair A→B: slope_end_A and slope_start_B.
  Grouped by transition type to study continuity at junctions.

Conditioning dimensions
-----------------------
  performer   : 3-letter code extracted from recording_id (bdn, drn, ...)
  tempo_bucket: slow / medium / fast — binned from median svara dur per recording
  dur_bin     : short / medium / long — binned per-segment duration (tertiles)

Outputs
-------
  figures/analysis/shape_stats/cp_concavity.png
  figures/analysis/shape_stats/sta_params.png
  figures/analysis/shape_stats/tr_params.png
  figures/analysis/shape_stats/sta_tr_aggregated.png
  figures/analysis/shape_stats/transitions_derivative.png

Usage
-----
    python -m src.analysis.shape_stats
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import settings as S
from src.analysis.svara_segment_analysis import COLORS as PERFORMER_COL, SCALE_ORDER as SVARA_ORDER
from src.models.curve_vae.fit_sta_tr_curves import _h

INTERIM_DIR = S.DATA_INTERIM / "analysis"
FIGURES_DIR = S.FIGURES_DIR / "analysis" / "shape_stats"

DUR_ORDER = ["short", "medium", "long"]


# ── data loading ──────────────────────────────────────────────────────────────

def _load_svara_durations() -> pl.DataFrame:
    dfs = []
    for rec in S.SARASUDA_VARNAM:
        p = S.DATA_INTERIM / rec / "features" / f"{rec}_svara_structural_embeddings.parquet"
        if p.exists():
            dfs.append(
                pl.read_parquet(p)
                .select(["recording_id", "segment_id", "duration_sec"])
                .rename({"segment_id": "svara_ann_idx", "duration_sec": "svara_dur_sec"})
            )
    return pl.concat(dfs)


def load_enriched() -> pl.DataFrame:
    df = pl.read_parquet(INTERIM_DIR / "segment_shapes.parquet")

    df = df.with_columns(
        pl.col("recording_id").str.split("_").list.get(2).alias("performer")
    )

    sv_dur = _load_svara_durations()
    df = df.join(sv_dur, on=["recording_id", "svara_ann_idx"], how="left")

    rec_tempo = (
        sv_dur.group_by("recording_id")
        .agg(pl.col("svara_dur_sec").median().alias("median_svara_dur"))
        .sort("median_svara_dur")
        .with_row_index("tempo_rank")
    )
    n = rec_tempo.height
    rec_tempo = rec_tempo.with_columns(
        pl.when(pl.col("tempo_rank") < n // 3).then(pl.lit("fast"))
        .when(pl.col("tempo_rank") < 2 * n // 3).then(pl.lit("medium"))
        .otherwise(pl.lit("slow"))
        .alias("tempo_bucket")
    )
    df = df.join(rec_tempo.select(["recording_id", "tempo_bucket"]), on="recording_id", how="left")

    dur_bin_col = pl.lit(None).cast(pl.String)
    for seg_type in ["CP", "STAp", "STAt", "TRa", "TRd"]:
        mask = pl.col("seg_type") == seg_type
        sub  = df.filter(mask)
        q33, q67 = sub["dur_sec"].quantile(0.33), sub["dur_sec"].quantile(0.67)
        dur_bin_col = (
            pl.when(mask & (pl.col("dur_sec") <= q33)).then(pl.lit("short"))
            .when(mask & (pl.col("dur_sec") <= q67)).then(pl.lit("medium"))
            .when(mask).then(pl.lit("long"))
            .otherwise(dur_bin_col)
        )
    df = df.with_columns(dur_bin_col.alias("dur_bin"))

    return df


# ── boundary slope computation ────────────────────────────────────────────────

def _boundary_slopes(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add slope_start (cents/s at t=0) and slope_end (cents/s at t=1) to every row.

    STA/TR : analytical derivative of normalised model × Δcents/dur_sec
    CP     : quadratic fit (concavity=a) → slope at t=0 and t=dur_sec
    SIL    : 0
    """
    seg   = df["seg_type"].to_numpy()
    dur   = np.maximum(df["dur_sec"].to_numpy().astype(float), 1e-6)
    sc    = np.nan_to_num(df["start_cents"].to_numpy().astype(float))
    ec    = np.nan_to_num(df["end_cents"].to_numpy().astype(float))
    delta = ec - sc

    k_arr = df["k_steep"].to_numpy().astype(float)
    s_arr = df["s_inflect"].to_numpy().astype(float)
    A_arr = df["A_osc"].to_numpy().astype(float)
    a_arr = df["concavity"].to_numpy().astype(float)  # may contain NaN

    slope_start = np.zeros(len(df))
    slope_end   = np.zeros(len(df))

    # STA / TR — vectorised analytical derivative
    # h(t) = tanh(k*(t-s)) + A*sin(2π*(t-0.5))
    # sin(2π*(0-0.5))=sin(-π)=0, sin(2π*(1-0.5))=sin(π)=0 → h0=tanh(-k*s), h1=tanh(k*(1-s))
    # h'(t) = k/cosh²(k*(t-s)) + 2πA*cos(2π*(t-0.5))
    # cos(2π*(0-0.5))=cos(-π)=-1, cos(2π*(1-0.5))=cos(π)=-1
    h0    = np.tanh(-k_arr * s_arr)
    h1    = np.tanh(k_arr * (1 - s_arr))
    denom = h1 - h0
    safe  = np.abs(denom) > 1e-9
    dh0   = k_arr / np.cosh(k_arr * s_arr) ** 2       - 2 * np.pi * A_arr
    dh1   = k_arr / np.cosh(k_arr * (1 - s_arr)) ** 2 - 2 * np.pi * A_arr
    nd0   = np.where(safe, dh0 / np.where(safe, denom, 1.0), 0.0)
    nd1   = np.where(safe, dh1 / np.where(safe, denom, 1.0), 0.0)

    sta_tr = np.isin(seg, ["STAp", "STAt", "TRa", "TRd"])
    valid_params = sta_tr & np.isfinite(k_arr) & np.isfinite(s_arr) & np.isfinite(A_arr)
    slope_start = np.where(valid_params, nd0 * delta / dur,
                  np.where(sta_tr,       delta / dur, slope_start))
    slope_end   = np.where(valid_params, nd1 * delta / dur,
                  np.where(sta_tr,       delta / dur, slope_end))

    # CP — quadratic: y = a*t² + b*t + c,  c=start,  y(dur)=end
    is_cp      = seg == "CP"
    finite_a   = is_cp & np.isfinite(a_arr)
    b_arr      = (delta - a_arr * dur ** 2) / dur
    slope_start = np.where(finite_a, b_arr,             np.where(is_cp, delta / dur, slope_start))
    slope_end   = np.where(finite_a, 2*a_arr*dur + b_arr, np.where(is_cp, delta / dur, slope_end))

    return df.with_columns([
        pl.Series("slope_start", slope_start),
        pl.Series("slope_end",   slope_end),
    ])


# ── transition pairs ──────────────────────────────────────────────────────────

def build_transition_pairs(df: pl.DataFrame) -> pl.DataFrame:
    records = []
    for (rec, svara_idx), sub in df.group_by(["recording_id", "svara_ann_idx"]):
        sub  = sub.sort("seg_idx")
        rows = sub.to_dicts()
        for j in range(len(rows) - 1):
            a = rows[j]
            b = rows[j + 1]
            records.append({
                "recording_id":  rec,
                "svara_label":   a["svara_label"],
                "performer":     a["performer"],
                "tempo_bucket":  a.get("tempo_bucket"),
                "transition":    f"{a['seg_type']}→{b['seg_type']}",
                "type_A":        a["seg_type"],
                "type_B":        b["seg_type"],
                "slope_end_A":   float(a["slope_end"]),
                "slope_start_B": float(b["slope_start"]),
                "kink":          float(a["slope_end"]) - float(b["slope_start"]),
            })
    return pl.DataFrame(records)


# ── plotting helpers ──────────────────────────────────────────────────────────

def _violin(ax: plt.Axes, data: dict[str, np.ndarray], title: str, ylabel: str,
            order: list[str] | None = None, colors: list | None = None) -> None:
    keys  = order if order else list(data.keys())
    arrs  = [data[k] for k in keys if k in data and len(data[k]) > 2]
    lbls  = [k for k in keys if k in data and len(data[k]) > 2]
    if not arrs:
        return
    parts = ax.violinplot(arrs, showmedians=True, showextrema=False)
    for j, body in enumerate(parts["bodies"]):
        body.set_alpha(0.6)
        if colors:
            body.set_facecolor(colors[j % len(colors)])
    parts["cmedians"].set_color("k")
    ax.set_xticks(range(1, len(lbls) + 1))
    ax.set_xticklabels(lbls, fontsize=7)
    ax.set_title(title, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=7)
    ax.tick_params(labelsize=6)
    ax.axhline(0, lw=0.5, ls="--", color="gray", alpha=0.5)


# ── CP analysis ───────────────────────────────────────────────────────────────

def plot_cp_concavity(df_cp: pl.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    performers = sorted(df_cp["performer"].drop_nulls().unique().to_list())
    panels = [
        (axes[0], {sv: df_cp.filter(pl.col("svara_label") == sv)["concavity"].drop_nulls().to_numpy()
                   for sv in SVARA_ORDER}, "CP concavity by svara", SVARA_ORDER, None),
        (axes[1], {b: df_cp.filter(pl.col("dur_bin") == b)["concavity"].drop_nulls().to_numpy()
                   for b in DUR_ORDER}, "CP concavity by duration", DUR_ORDER, None),
        (axes[2], {p: df_cp.filter(pl.col("performer") == p)["concavity"].drop_nulls().to_numpy()
                   for p in performers}, "CP concavity by performer", performers,
         [PERFORMER_COL.get(p, "#888") for p in performers]),
    ]
    for ax, data, title, order, colors in panels:
        _violin(ax, data, title, "concavity (¢/s²)", order, colors)

    fig.suptitle("CP concavity analysis", fontsize=10, fontweight="bold")
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {out}")


# ── STA / TR parameter analysis ───────────────────────────────────────────────

def _plot_params_panel(df: pl.DataFrame, seg_type: str, out: Path) -> None:
    params     = [("k_steep", "k (steepness)"), ("s_inflect", "s (inflection)"), ("A_osc", "A (oscillation)")]
    performers = sorted(df["performer"].drop_nulls().unique().to_list())
    colors_p   = [PERFORMER_COL.get(p, "#888") for p in performers]

    fig, axes = plt.subplots(3, 3, figsize=(13, 10))

    for row, (col, ylabel) in enumerate(params):
        panels = [
            (axes[row][0], {sv: df.filter(pl.col("svara_label") == sv)[col].drop_nulls().to_numpy()
                            for sv in SVARA_ORDER}, f"{col} by svara", SVARA_ORDER, None),
            (axes[row][1], {b: df.filter(pl.col("dur_bin") == b)[col].drop_nulls().to_numpy()
                            for b in DUR_ORDER}, f"{col} by duration", DUR_ORDER, None),
            (axes[row][2], {p: df.filter(pl.col("performer") == p)[col].drop_nulls().to_numpy()
                            for p in performers}, f"{col} by performer", performers, colors_p),
        ]
        for ax, data, title, order, colors in panels:
            _violin(ax, data, title, ylabel, order, colors)

    fig.suptitle(f"{seg_type} parameter distributions", fontsize=10, fontweight="bold")
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {out}")


def plot_sta_tr_aggregated(df_a: pl.DataFrame, df_b: pl.DataFrame, out: Path) -> None:
    label_a = df_a["seg_type"].head(1).to_list()[0] if df_a.height > 0 else "A"
    label_b = df_b["seg_type"].head(1).to_list()[0] if df_b.height > 0 else "B"
    df = pl.concat([
        df_a.with_columns(pl.lit(label_a).alias("seg_type_label")),
        df_b.with_columns(pl.lit(label_b).alias("seg_type_label")),
    ])
    params = [("k_steep", "k"), ("s_inflect", "s"), ("A_osc", "A")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (col, ylabel) in zip(axes, params):
        data = {t: df.filter(pl.col("seg_type_label") == t)[col].drop_nulls().to_numpy()
                for t in [label_a, label_b]}
        _violin(ax, data, f"{col} {label_a} vs {label_b}", ylabel, [label_a, label_b])
    fig.suptitle(f"{label_a} vs {label_b} parameters", fontsize=10, fontweight="bold")
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {out}")


# ── inter-segment derivative analysis ─────────────────────────────────────────

def plot_transitions(df_trans: pl.DataFrame, out: Path) -> None:
    transitions = (
        df_trans["transition"].value_counts()
        .sort("count", descending=True)
        .filter(pl.col("count") >= 10)
        ["transition"].to_list()
    )
    if not transitions:
        return

    n   = len(transitions)
    fig = plt.figure(figsize=(14, 4 * ((n + 2) // 3)))
    gs  = gridspec.GridSpec((n + 2) // 3, 3, figure=fig)

    for idx, trans in enumerate(transitions):
        sub   = df_trans.filter(pl.col("transition") == trans)
        end_A = sub["slope_end_A"].to_numpy()
        st_B  = sub["slope_start_B"].to_numpy()
        kink  = sub["kink"].to_numpy()

        ax   = fig.add_subplot(gs[idx // 3, idx % 3])
        lo   = min(end_A.min(), st_B.min())
        hi   = max(end_A.max(), st_B.max())
        bins = np.linspace(lo, hi, 40)
        ax.hist(end_A, bins=bins, alpha=0.5, label="slope end A",   color="#e41a1c", density=True)
        ax.hist(st_B,  bins=bins, alpha=0.5, label="slope start B", color="#377eb8", density=True)
        ax.set_title(f"{trans}  (n={sub.height})", fontsize=8)
        ax.set_xlabel("¢/s", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.axvline(0, lw=0.5, ls="--", color="gray")
        if idx == 0:
            ax.legend(fontsize=6)

        ax2 = ax.inset_axes([0.62, 0.55, 0.36, 0.38])
        ax2.hist(kink, bins=30, color="#4daf4a", alpha=0.7, density=True)
        ax2.axvline(0, lw=0.8, ls="--", color="gray")
        ax2.set_title("kink", fontsize=6)
        ax2.tick_params(labelsize=5)

    fig.suptitle("Inter-segment derivative analysis", fontsize=10, fontweight="bold")
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {out}")


def plot_kink_by_svara(df_trans: pl.DataFrame, out: Path) -> None:
    transitions = (
        df_trans["transition"].value_counts()
        .sort("count", descending=True)
        .filter(pl.col("count") >= 10)
        ["transition"].to_list()
    )

    fig, axes = plt.subplots(1, len(transitions), figsize=(4 * len(transitions), 4), squeeze=False)

    for ax, trans in zip(axes[0], transitions):
        sub    = df_trans.filter(pl.col("transition") == trans)
        groups = {
            row["svara_label"]: np.array(row["kink"])
            for row in sub.group_by("svara_label").agg(pl.col("kink")).iter_rows(named=True)
            if len(row["kink"]) > 2
        }
        data_sv = {sv: groups[sv] for sv in SVARA_ORDER if sv in groups}
        _violin(ax, data_sv, trans, "kink (¢/s)", list(data_sv.keys()))

    fig.suptitle("Kink by svara for each transition type", fontsize=10, fontweight="bold")
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("[shape_stats] loading data...")
    df = load_enriched()
    df = _boundary_slopes(df)

    df_cp   = df.filter(pl.col("seg_type") == "CP")
    df_stap = df.filter(pl.col("seg_type") == "STAp")
    df_stat = df.filter(pl.col("seg_type") == "STAt")
    df_tra  = df.filter(pl.col("seg_type") == "TRa")
    df_trd  = df.filter(pl.col("seg_type") == "TRd")

    print(f"  CP={df_cp.height}  STAp={df_stap.height}  STAt={df_stat.height}  "
          f"TRa={df_tra.height}  TRd={df_trd.height}")

    print("[shape_stats] building transition pairs...")
    df_trans = build_transition_pairs(df)
    print(f"  transitions={df_trans.height}  types={df_trans['transition'].n_unique()}")
    print(df_trans["transition"].value_counts().sort("count", descending=True))

    print("[shape_stats] plotting...")
    plot_cp_concavity(df_cp, FIGURES_DIR / "cp_concavity.png")
    _plot_params_panel(df_stap, "STAp", FIGURES_DIR / "stap_params.png")
    _plot_params_panel(df_stat, "STAt", FIGURES_DIR / "stat_params.png")
    _plot_params_panel(df_tra,  "TRa",  FIGURES_DIR / "tra_params.png")
    _plot_params_panel(df_trd,  "TRd",  FIGURES_DIR / "trd_params.png")
    plot_sta_tr_aggregated(df_stap, df_trd, FIGURES_DIR / "stap_trd_aggregated.png")
    plot_sta_tr_aggregated(df_stat, df_tra, FIGURES_DIR / "stat_tra_aggregated.png")
    plot_transitions(df_trans, FIGURES_DIR / "transitions_derivative.png")
    plot_kink_by_svara(df_trans, FIGURES_DIR / "kink_by_svara.png")

    print("[shape_stats] done.")


if __name__ == "__main__":
    main()
