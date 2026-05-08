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
from dataclasses import fields as dataclass_fields

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import settings
from src.models.gruvae.dataset_gruvae import SVARA_LABELS, SVARA_TO_IDX
from src.models.gruvae.model_gruvae import ModelConfig, SvaraGRUVAE

CKPT_DIR   = settings.DATA_INTERIM / "models" / "gruvae_v4"
TYPE_NAMES = ["CP", "SIL", "STA", "TR"]
COLOR      = {"CP": "#4caf50", "SIL": "#9e9e9e", "STA": "#ff9800", "TR": "#2196f3"}
ALPHA      = {"CP": 0.40,      "SIL": 0.30,      "STA": 0.35,      "TR": 0.30}
BOX_H      = 50   # half-height of each box in cents


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(run: str | None, device: torch.device) -> SvaraGRUVAE:
    run_dir = (CKPT_DIR / run) if run else sorted(CKPT_DIR.glob("run_*"))[-1]
    best    = run_dir / "best.pt"
    if not best.exists():
        best = run_dir / "last.pt"
    ckpt      = torch.load(best, map_location=device)
    known     = {f.name for f in dataclass_fields(ModelConfig)}
    cfg_dict  = {k: v for k, v in ckpt["model_cfg"].items() if k in known}
    cfg_dict.setdefault("use_attention", False)   # older checkpoints lack this key
    cfg       = ModelConfig(**cfg_dict)
    model = SvaraGRUVAE(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[gruvae] epoch={ckpt['epoch']}  best_val={ckpt['best_val']:.4f}  {best}")
    return model


# ── generation ────────────────────────────────────────────────────────────────

def generate(
    model:         SvaraGRUVAE,
    svara_label:   str,
    n:             int,
    device:        torch.device,
    use_hard_mask: bool = False,
) -> list[dict]:
    idx_t = torch.tensor(
        [SVARA_TO_IDX[svara_label]] * n, dtype=torch.long, device=device
    )
    with torch.no_grad():
        out      = model.generate(batch_size=n, svara_idx=idx_t, device=device,
                                  use_hard_mask=use_hard_mask)
    gen       = out["generated"]         # (n, max_len, 6): [CP, SIL, STA, TR, dur_rel, cents_norm]
    pred_lens = out["pred_length"]       # (n,)

    results = []
    for i in range(n):
        n_segs = int(pred_lens[i].round().clamp(1, model.cfg.max_seq_len).item())
        segs   = gen[i, :n_segs].cpu().numpy()

        segments = []
        type_dim = model.cfg.type_dim   # 4: [CP, SIL, STA, TR]
        for s in segs:
            typ = TYPE_NAMES[int(np.argmax(s[:type_dim]))]
            cents = 0.0 if typ in ("SIL", "TR") else float(s[type_dim + 1]) * 1200.0
            segments.append({
                "type":    typ,
                "dur_rel": float(s[type_dim]),
                "cents":   cents,
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

        if typ in ("CP", "STA"):
            ax.hlines(cents, x, x + w, colors=COLOR[typ], lw=1.8, zorder=3)
            label = f"{typ}  {cents:+.0f}¢"
        elif typ == "SIL":
            ax.hlines(0, x, x + w, colors=COLOR[typ], lw=1.8, ls="--", zorder=3)
            label = "SIL"
        else:  # TR
            ax.hlines(0, x, x + w, colors=COLOR[typ], lw=1.8, ls=":", zorder=3)
            label = "TR"

        text_y = cents if typ in ("CP", "STA") else 0
        ax.text(
            x + w / 2, text_y + BOX_H * 0.52,
            label,
            ha="center", va="bottom",
            fontsize=6.5, fontweight="bold",
            color="black",
        )
        ax.text(
            x + w / 2, text_y - BOX_H * 0.52,
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
    parser.add_argument("--hard-mask", action="store_true",
                        help="Enforce musical grammar during generation (type transition mask)")
    args = parser.parse_args()

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model    = load_model(args.run, device)
    run_name = args.run if args.run else sorted(CKPT_DIR.glob("run_*"))[-1].name

    svaras = SVARA_LABELS if args.all else [args.svara]
    n_rows = len(svaras)
    n_cols = args.n

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 3.5, n_rows * 3.0),
        squeeze=False,
    )
    mask_tag  = "_hardmask" if args.hard_mask else ""
    fig.suptitle(
        f"SvaraGRUVAE [{run_name}]{' [hard-mask]' if args.hard_mask else ''} — generated svara structures",
        fontsize=11, fontweight="bold",
    )

    for r, svara in enumerate(svaras):
        samples = generate(model, svara, args.n, device, use_hard_mask=args.hard_mask)
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

    svara_tag = "all" if args.all else args.svara
    out_path  = Path(args.out) if args.out else \
                settings.FIGURES_DIR / "gruvae" / run_name / f"generated_{svara_tag}{mask_tag}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[gruvae] saved → {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
