"""
Compare generated parametric curves against GT for STA and TR segments.

Outputs:
    figures/curve_vae/generated_vs_gt.png

Usage:
    python -m src.models.curve_vae.plot_generated
    python -m src.models.curve_vae.plot_generated --svara G --n 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import settings as S
from src.models.curve_vae.sta_tr_model import CurveModel, SVARA_LABELS

FITTED  = S.CURVE_VAE_DIR / "gt_curves_fitted.parquet"
OUT_DIR = S.FIGURES_DIR / "curve_vae"
T_DENSE = np.linspace(0, 1, 200)


def plot_comparison(
    df: pl.DataFrame,
    model: CurveModel,
    svara: str | None,
    n_gt: int,
    n_gen: int,
    out: Path,
) -> None:
    rng = np.random.default_rng(1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)

    _GROUP = [
        (axes[0], "STA", ["STAp", "STAt"], "#ff9800", "#b71c1c", (-0.15, 1.25)),
        (axes[1], "TR",  ["TRa",  "TRd"],  "#2196f3", "#1a237e", (-0.6,  1.4)),
    ]
    for ax, label, subtypes, color_gt, color_gen, ylim in _GROUP:
        sub = df.filter(pl.col("seg_type").is_in(subtypes))
        if svara:
            sub = sub.filter(pl.col("svara_label") == svara)
        if len(sub) == 0:
            ax.set_title(f"{label} (no data)")
            continue

        n_show = min(n_gt, len(sub))
        sample = sub.sample(n_show, seed=2, with_replacement=False)

        # plot GT curves
        for i, row in enumerate(sample.iter_rows(named=True)):
            t = np.array(row["t_norm"])
            p = np.array(row["p_norm"])
            lbl = "GT" if i == 0 else None
            ax.plot(t, p, "o-", ms=2, lw=0.8, color=color_gt, alpha=0.5, label=lbl)

        # plot generated curves (split evenly across subtypes)
        n_each = max(1, n_gen // len(subtypes))
        gen_labeled = False
        for st in subtypes:
            for g in model.generate(st, svara=svara, n=n_each, rng=rng):
                y   = g["curve_fn"](T_DENSE)
                lbl = "Generated" if not gen_labeled else None
                gen_labeled = True
                ax.plot(T_DENSE, y, "-", lw=1.5, color=color_gen, alpha=0.65, label=lbl)

        sv_tag = f" (svara={svara})" if svara else " (all svaras)"
        ax.set_title(f"{label}{sv_tag}  n_gt={n_show}  n_gen={n_gen}", fontsize=9)
        ax.set_xlabel("t (normalized)", fontsize=8)
        ax.set_ylabel("pitch (normalized)", fontsize=8)
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_ylim(*ylim)

    sv_title = f"svara={svara}" if svara else "all svaras"
    fig.suptitle(f"Generated vs GT pitch curves  [{sv_title}]",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  → {out}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--svara", default=None, choices=SVARA_LABELS + [None])
    parser.add_argument("--n",     type=int, default=30, help="GT curves to show")
    parser.add_argument("--n-gen", type=int, default=20, help="Generated curves to show")
    parser.add_argument("--out",   default=None)
    args = parser.parse_args()

    df    = pl.read_parquet(FITTED)
    model = CurveModel().fit(df)

    sv_tag = args.svara or "all"
    out = Path(args.out) if args.out else OUT_DIR / f"generated_vs_gt_{sv_tag}.png"

    plot_comparison(df, model, args.svara, n_gt=args.n, n_gen=args.n_gen, out=out)


if __name__ == "__main__":
    main()
