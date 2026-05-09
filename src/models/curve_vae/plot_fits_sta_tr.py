"""
Visualize fitted parametric curves and parameter distributions.

Outputs:
    figures/curve_vae/fit_quality.png     — sample fits (GT vs model)
    figures/curve_vae/param_dists.png     — k, s, A distributions
    figures/curve_vae/param_scatter.png   — k vs s, coloured by A

Usage:
    python -m src.models.curve_vae.plot_fits
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import settings as S
from src.models.curve_vae.fit_sta_tr_curves import curve_model, curve_model_tr
from src.models.curve_vae.sta_tr_model import CurveModel
from src.models.curve_vae.plot_generated_sta_tr import plot_comparison

FITTED  = S.DATA_INTERIM / "models" / "curve_vae" / "gt_curves_fitted.parquet"
OUT_DIR = S.FIGURES_DIR / "curve_vae"

SVARA_LABELS = sorted(['D', 'G', 'M', 'N', 'P', 'R', 'S'])
T_DENSE      = np.linspace(0, 1, 200)


def plot_sample_fits(df: pl.DataFrame, out: Path, n_sta: int = 12, n_tr: int = 8) -> None:
    # sample: mix of normal and N-shape curves
    sta_n = df.filter((pl.col("seg_type") == "STA") & (pl.col("A_osc") < -0.15)).sample(
        min(4, int((df["seg_type"] == "STA").sum())), seed=1)
    sta_r = df.filter((pl.col("seg_type") == "STA") & (pl.col("A_osc") >= -0.15)).sample(
        n_sta - len(sta_n), seed=42)
    tr_n  = df.filter((pl.col("seg_type") == "TR")  & (pl.col("A_osc") < -0.15)).sample(
        min(3, int((df["seg_type"] == "TR").sum())), seed=1)
    tr_r  = df.filter((pl.col("seg_type") == "TR")  & (pl.col("A_osc") >= -0.15)).sample(
        n_tr - len(tr_n), seed=42)
    sta = pl.concat([sta_n, sta_r])
    tr  = pl.concat([tr_n,  tr_r])

    ncols = 4
    total = n_sta + n_tr
    nrows = (total + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 2.2), squeeze=False)
    axes_flat = axes.ravel()

    ax_idx = 0
    for seg_df, fn, clr in [(sta, curve_model, "r"), (tr, curve_model_tr, "b")]:
        for row in seg_df.iter_rows(named=True):
            ax  = axes_flat[ax_idx]; ax_idx += 1
            t   = np.array(row["t_norm"])
            p   = np.array(row["p_norm"])
            k, s, A = row["k_steep"], row["s_inflect"], row["A_osc"]
            yh  = fn(T_DENSE, k, s, A)
            tag = " [N]" if A < -0.15 else ""
            ax.plot(t, p, "ko", ms=3, label="GT")
            ax.plot(T_DENSE, yh, f"{clr}-", lw=1.5,
                    label=f"k={k:.1f} s={s:.2f} A={A:+.2f}{tag}")
            ax.set_title(f"{row['seg_type']}  R²={row['r2']:.4f}", fontsize=7)
            ax.set_ylim(-0.25, 1.25)
            ax.legend(fontsize=5.5); ax.tick_params(labelsize=6)

    for ax in axes_flat[ax_idx:]:
        ax.set_visible(False)

    fig.suptitle("tanh+osc fits  (GT vs model)  — [N] = N-shape",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  → {out}")
    plt.close(fig)


def plot_param_distributions(df: pl.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))

    for row_i, (seg_type, clr) in enumerate([("STA", "#ff9800"), ("TR", "#2196f3")]):
        sub = df.filter(pl.col("seg_type") == seg_type)
        for col_i, (arr, title) in enumerate([
            (sub["k_steep"].to_numpy(),   f"{seg_type} — k (steepness)"),
            (sub["s_inflect"].to_numpy(),  f"{seg_type} — s (skew / inflection)"),
            (sub["A_osc"].to_numpy(),      f"{seg_type} — A (oscillation / N-shape)"),
        ]):
            ax = axes[row_i, col_i]
            ax.hist(arr, bins=50, color=clr, alpha=0.8)
            ax.axvline(float(np.median(arr)), color="k", ls="--", lw=1.2,
                       label=f"med={np.median(arr):.3f}")
            if col_i == 2:   # A column: mark N-shape threshold
                ax.axvline(-0.15, color="r", ls=":", lw=1.0, label="N-shape thr")
                n_n = int((arr < -0.15).sum())
                ax.set_title(f"{title}  (N-shape {n_n}/{len(arr)})", fontsize=8)
            else:
                ax.set_title(title, fontsize=9)
            ax.legend(fontsize=7)
            ax.tick_params(labelsize=7)

    fig.suptitle("tanh+osc parameter distributions", fontsize=11, fontweight="bold")
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  → {out}")
    plt.close(fig)


def plot_ks_scatter(df: pl.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, seg_type in zip(axes, ("STA", "TR")):
        sub = df.filter(pl.col("seg_type") == seg_type)
        k   = sub["k_steep"].to_numpy()
        s   = sub["s_inflect"].to_numpy()
        A   = sub["A_osc"].to_numpy()
        sc  = ax.scatter(k, s, c=A, cmap="RdYlGn", s=6, alpha=0.4,
                         vmin=-0.5, vmax=0.5)
        plt.colorbar(sc, ax=ax, label="A (green=0, red=N-shape)")
        ax.axhline(0.5, color="gray", ls=":", lw=0.8, label="s=0.5")
        ax.axvline(float(np.median(k)), color="k", ls="--", lw=0.8)
        ax.axhline(float(np.median(s)), color="k", ls="--", lw=0.8)
        ax.set_xlabel("k (steepness)", fontsize=8)
        ax.set_ylabel("s (inflection)", fontsize=8)
        ax.set_title(f"{seg_type} — k vs s  (colour=A)", fontsize=9)
        ax.legend(fontsize=7); ax.tick_params(labelsize=7)

    fig.suptitle("tanh+osc parameter scatter", fontsize=11, fontweight="bold")
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  → {out}")
    plt.close(fig)


def main() -> None:
    df    = pl.read_parquet(FITTED)
    model = CurveModel().fit(df)
    print(f"Loaded {len(df)} fitted curves")
    plot_sample_fits(df,         OUT_DIR / "fit_quality.png")
    plot_param_distributions(df, OUT_DIR / "param_dists.png")
    plot_ks_scatter(df,          OUT_DIR / "param_scatter.png")
    plot_comparison(df, model, svara=None, n_gt=30, n_gen=20,
                    out=OUT_DIR / "generated_vs_gt_all.png")


if __name__ == "__main__":
    main()
