"""
Generate svaras from SvaraGRUVAE and plot structural diagrams.

Each generated svara is shown as coloured segment boxes:
    green  = CP  (stable pitch position)
    orange = STA (staircase / transient)
    gray   = SIL (silence)

Box width  ∝ dur_rel  (normalised duration)
Box centre Y = cents  (cents relative to tonic, from cents_norm × 1200)

Labels inside each box: type / cents value / proportional duration %.

Usage:
    python -m src.models.gruvae.generate_plot
    python -m src.models.gruvae.generate_plot --svara S --n 5
    python -m src.models.gruvae.generate_plot --all --n 3
    python -m src.models.gruvae.generate_plot --svara G --n 4 --out figures/gen_G.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import settings
from src.models.gruvae.dataset_gruvae import SVARA_LABELS, SVARA_TO_IDX
from src.models.gruvae.model_gruvae import ModelConfig, SvaraGRUVAE

CKPT_DIR   = settings.DATA_INTERIM / "models" / "gruvae_v4"
TYPE_NAMES = ["CP", "SIL", "STA"]
COLOR      = {"CP": "#4caf50", "SIL": "#9e9e9e", "STA": "#ff9800"}
ALPHA      = {"CP": 0.40,      "SIL": 0.30,      "STA": 0.35}
BOX_H      = 50   # half-height of each box in cents


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(run: str | None, device: torch.device) -> SvaraGRUVAE:
    run_dir = (CKPT_DIR / run) if run else sorted(CKPT_DIR.glob("run_*"))[-1]
    best    = run_dir / "best.pt"
    if not best.exists():
        best = run_dir / "last.pt"
    ckpt  = torch.load(best, map_location=device)
    cfg   = ModelConfig(**ckpt["model_cfg"])
    model = SvaraGRUVAE(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[gruvae] epoch={ckpt['epoch']}  best_val={ckpt['best_val']:.4f}  {best}")
    return model


# ── generation ────────────────────────────────────────────────────────────────

def generate(
    model:       SvaraGRUVAE,
    svara_label: str,
    n:           int,
    device:      torch.device,
) -> list[dict]:
    idx_t = torch.tensor(
        [SVARA_TO_IDX[svara_label]] * n, dtype=torch.long, device=device
    )
    with torch.no_grad():
        out      = model.generate(batch_size=n, svara_idx=idx_t, device=device)
    gen       = out["generated"]         # (n, max_len, 5): [CP, SIL, STA, dur_rel, cents_norm]
    pred_lens = out["pred_length"]       # (n,)

    results = []
    for i in range(n):
        n_segs = int(pred_lens[i].round().clamp(1, model.cfg.max_seq_len).item())
        segs   = gen[i, :n_segs].cpu().numpy()

        segments = []
        for s in segs:
            segments.append({
                "type":    TYPE_NAMES[int(np.argmax(s[:3]))],
                "dur_rel": float(s[3]),
                "cents":   float(s[4]) * 1200.0,
            })
        results.append({"svara": svara_label, "segments": segments})
    return results


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_svara(ax: plt.Axes, data: dict, title: str = "") -> None:
    segs      = data["segments"]
    svara     = data["svara"]
    n_segs    = len(segs)
    total_rel = sum(s["dur_rel"] for s in segs) or 1.0

    x = 0.0
    for seg in segs:
        w     = seg["dur_rel"] / total_rel
        cents = seg["cents"]
        typ   = seg["type"]

        rect = mpatches.FancyBboxPatch(
            (x, cents - BOX_H), w, 2 * BOX_H,
            boxstyle="round,pad=0.005",
            linewidth=1.0,
            edgecolor=COLOR[typ],
            facecolor=COLOR[typ],
            alpha=ALPHA[typ],
        )
        ax.add_patch(rect)
        rect.set_clip_path(ax.patch)

        # horizontal pitch line (dashed for SIL)
        ls = "--" if typ == "SIL" else "-"
        ax.hlines(cents, x, x + w, colors=COLOR[typ], lw=1.8, ls=ls, zorder=3)

        # text in the upper quarter of the box so it clears the line
        ax.text(
            x + w / 2, cents + BOX_H * 0.52,
            f"{typ}  {cents:+.0f}¢",
            ha="center", va="bottom",
            fontsize=6.5, fontweight="bold",
            color="black",
        )
        ax.text(
            x + w / 2, cents - BOX_H * 0.52,
            f"{seg['dur_rel'] * 100:.1f}%",
            ha="center", va="top",
            fontsize=6, color="black",
        )
        x += w

    # reference lines
    ax.axhline(0,    color="steelblue", lw=0.6, ls="--", alpha=0.6, label="tonic (0¢)")
    ax.axhline(1200, color="steelblue", lw=0.6, ls=":",  alpha=0.4, label="octave (1200¢)")

    all_cents = [s["cents"] for s in segs]
    pad       = 120
    ax.set_xlim(0.0, 1.0)
    y_lo = max(min(all_cents) - pad, -200)
    y_hi = min(max(all_cents) + pad, 1400)
    ax.set_ylim(y_lo, y_hi)
    ax.set_ylabel("Cents (rel. tonic)", fontsize=7)
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    ax.tick_params(axis="y", labelsize=7)

    dur_str = f"Σdur_rel={total_rel:.2f}  n_segs={n_segs}"
    ax.set_title(
        title or f"Svara {svara}  ({n_segs} segs)  {dur_str}",
        fontsize=8,
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and plot svaras from SvaraGRUVAE."
    )
    parser.add_argument("--svara", default="S", choices=SVARA_LABELS,
                        help="Svara label to condition on")
    parser.add_argument("--all",   action="store_true",
                        help="Generate all 7 svara labels")
    parser.add_argument("--n",     type=int, default=4,
                        help="Samples per svara label")
    parser.add_argument("--run",   default=None,
                        help="Run directory, e.g. run_001 (default: latest)")
    parser.add_argument("--out",   default=None,
                        help="Output PNG path (default: data/interim/gruvae_generated.png)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(args.run, device)

    svaras = SVARA_LABELS if args.all else [args.svara]
    n_rows = len(svaras)
    n_cols = args.n

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 3.5, n_rows * 3.0),
        squeeze=False,
    )
    fig.suptitle("SvaraGRUVAE — generated svara structures", fontsize=11, fontweight="bold")

    for r, svara in enumerate(svaras):
        samples = generate(model, svara, args.n, device)
        for c, sv_data in enumerate(samples):
            plot_svara(axes[r][c], sv_data, title=f"{svara} #{c + 1}")

    # shared legend
    handles = [
        mpatches.Patch(facecolor=COLOR[t], alpha=ALPHA[t]+0.2, label=t)
        for t in TYPE_NAMES
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    out_path = Path(args.out) if args.out else \
               settings.FIGURES_DIR / "gruvae" / "generated.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[gruvae] saved → {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
