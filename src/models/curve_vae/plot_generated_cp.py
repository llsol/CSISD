"""
Compare generated CP micro-variation curves against GT.

GT curves: pitch deviation in cents (mean removed, resampled to 64 samples).
Generated: sampled from CPVAE prior z ~ N(0,I), conditioned on svara + duration.

Outputs:
    figures/curve_vae/cp_generated_vs_gt.png          — all 7 svaras
    figures/curve_vae/cp_generated_vs_gt_{svara}.png  — single svara (--svara)

Usage:
    python -m src.models.curve_vae.plot_generated_cp
    python -m src.models.curve_vae.plot_generated_cp --svara G --n 10 --n-gen 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import settings as S
from src.models.gruvae.dataset_gruvae import SVARA_TO_IDX, SVARA_LABELS
from src.models.curve_vae.cp_vae import CPVAE, CPVAEConfig

DATA_PATH = S.DATA_INTERIM / "models" / "curve_vae" / "gt_cp_curves.parquet"
CKPT_DIR  = S.DATA_INTERIM / "models" / "curve_vae" / "cp_vae_runs"
OUT_DIR   = S.FIGURES_DIR / "curve_vae"


def load_model(device: torch.device) -> tuple[CPVAE, float]:
    run_dir = sorted(CKPT_DIR.glob("run_*"))[-1]
    best    = run_dir / "best.pt"
    if not best.exists():
        best = run_dir / "last.pt"
    ckpt  = torch.load(best, map_location=device)
    cfg   = CPVAEConfig(**ckpt["cfg"])
    model = CPVAE(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    scale = float(ckpt["scale"])
    print(f"[cp_vae] {run_dir.name}  epoch={ckpt['epoch']}  "
          f"best_val={ckpt['best_val']:.4f}  scale={scale:.2f}¢")
    return model, scale


def plot_all_svaras(
    df: pl.DataFrame,
    model: CPVAE,
    scale: float,
    device: torch.device,
    n_gt: int,
    n_gen: int,
    dur: float,
    out: Path,
) -> None:
    ncols = 4
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 2.8), squeeze=False)
    axes_flat = axes.ravel()

    for i, sv in enumerate(SVARA_LABELS):
        ax = axes_flat[i]
        _plot_one(ax, df, model, scale, device, sv, n_gt, n_gen, dur)

    axes_flat[len(SVARA_LABELS)].set_visible(False)

    legend_handles = [
        Line2D([0], [0], color="gray",    lw=1.2, alpha=0.6, label=f"GT (n={n_gt})"),
        Line2D([0], [0], color="#e65100", lw=1.2, alpha=0.7, label=f"Generated (n={n_gen})"),
        Line2D([0], [0], color="steelblue", lw=0.8, ls="--", label="zero (median pitch)"),
    ]
    fig.legend(handles=legend_handles, loc="lower right",
               bbox_to_anchor=(0.98, 0.01), fontsize=8)
    fig.suptitle(
        f"CP micro-variation VAE — GT vs Generated  (dur={dur:.2f}s conditioning)",
        fontsize=10, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.0, 1, 1])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  → {out}")
    plt.close(fig)


def plot_single_svara(
    df: pl.DataFrame,
    model: CPVAE,
    scale: float,
    device: torch.device,
    svara: str,
    n_gt: int,
    n_gen: int,
    dur: float,
    out: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    _plot_one(ax, df, model, scale, device, svara, n_gt, n_gen, dur)
    legend_handles = [
        Line2D([0], [0], color="gray",    lw=1.2, alpha=0.6, label=f"GT (n={n_gt})"),
        Line2D([0], [0], color="#e65100", lw=1.2, alpha=0.7, label=f"Generated (n={n_gen})"),
        Line2D([0], [0], color="steelblue", lw=0.8, ls="--", label="zero (median pitch)"),
    ]
    ax.legend(handles=legend_handles, fontsize=8)
    fig.suptitle(
        f"CP micro-variation VAE — svara {svara}  GT vs Generated  (dur={dur:.2f}s)",
        fontsize=10, fontweight="bold",
    )
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  → {out}")
    plt.close(fig)


def _plot_one(
    ax: plt.Axes,
    df: pl.DataFrame,
    model: CPVAE,
    scale: float,
    device: torch.device,
    svara: str,
    n_gt: int,
    n_gen: int,
    dur: float,
) -> None:
    sub = df.filter(pl.col("svara_label") == svara)
    n_show = min(n_gt, len(sub))

    # GT curves
    sample = sub.sample(n_show, seed=42, with_replacement=False)
    for row in sample.iter_rows(named=True):
        ax.plot(np.array(row["curve"]), color="gray", alpha=0.45, lw=0.8)

    # Generated curves
    sv_idx = torch.tensor([SVARA_TO_IDX[svara]] * n_gen, dtype=torch.long,  device=device)
    dur_t  = torch.tensor([dur]                 * n_gen, dtype=torch.float32, device=device)
    with torch.no_grad():
        gen = model.generate(n_gen, sv_idx, dur_t, scale=scale, verbose=True).cpu().numpy() * scale

    for g in gen:
        ax.plot(g, color="#e65100", alpha=0.65, lw=1.0)

    ax.axhline(0, color="steelblue", lw=0.7, ls="--", alpha=0.7)
    ax.set_title(f"CP  svara={svara}  (n_gt={n_show})", fontsize=8)
    ax.set_ylabel("deviation (¢)", fontsize=7)
    ax.set_ylim(-28, 28)
    ax.tick_params(labelsize=6)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot CP VAE generated curves vs GT."
    )
    parser.add_argument("--svara",  default=None, choices=SVARA_LABELS,
                        help="Single svara (default: all 7)")
    parser.add_argument("--n",      type=int, default=8,
                        help="GT curves per panel")
    parser.add_argument("--n-gen",  type=int, default=8,
                        help="Generated curves per panel")
    parser.add_argument("--dur",    type=float, default=0.18,
                        help="Duration conditioning in seconds (default: corpus mean)")
    parser.add_argument("--out",    default=None)
    args = parser.parse_args()

    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, scale = load_model(device)
    df           = pl.read_parquet(DATA_PATH)

    if args.svara:
        out = Path(args.out) if args.out else OUT_DIR / f"cp_generated_vs_gt_{args.svara}.png"
        plot_single_svara(df, model, scale, device,
                          args.svara, args.n, args.n_gen, args.dur, out)
    else:
        out = Path(args.out) if args.out else OUT_DIR / "cp_generated_vs_gt.png"
        plot_all_svaras(df, model, scale, device,
                        args.n, args.n_gen, args.dur, out)


if __name__ == "__main__":
    main()
