"""
Comparison plots: FTANet vs SwiftF0-scratch on SCMS test set (original audio).

Plots:
    1. Grouped bar chart — aggregate metrics (VR, VFA, RPA, RCA, OA)
    2. Per-stem scatter  — FTANet OA vs scratch OA (colour = scratch wins/loses)
    3. VFA histogram     — per-stem VFA distribution (bleeding hypothesis)

Usage:
    python -m src.pitch_extraction.swiftf0_scratch.plot_comparison
    python -m src.pitch_extraction.swiftf0_scratch.plot_comparison --thr 0.75 --run-scratch run_003
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import mir_eval

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import settings
from src.pitch_extraction.swiftf0_finetune.dataset import scms_official_split
from src.pitch_extraction.swiftf0_scratch.evaluate_scms import (
    load_annotation, load_ftanet, load_swiftf0,
    infer_swiftf0, swiftf0_audio_path, apply_threshold,
)
from src.pitch_extraction.swiftf0_scratch.model import SR as SR_SC, HOP as HOP_SC

SCMS_ROOT  = settings.PROJECT_ROOT / "data" / "datasets" / "scms"
PITCH_ROOT = settings.DATA_INTERIM / "scms_pitch"
OUT_BASE   = settings.FIGURES_DIR / "swiftf0_scratch" / "evaluation"

METRICS    = ["Voicing Recall", "Voicing False Alarm",
              "Raw Pitch Accuracy", "Raw Chroma Accuracy", "Overall Accuracy"]
LABELS     = ["VR", "VFA", "RPA", "RCA", "OA"]
COLORS     = {"FTANet": "#1f77b4", "SwiftF0-scratch": "#d62728"}


# ── per-stem evaluation ───────────────────────────────────────────────────────

def eval_per_stem(ref_t, ref_f, est_t, est_f) -> dict | None:
    try:
        return mir_eval.melody.evaluate(ref_t, ref_f, est_t, est_f)
    except Exception:
        return None


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_bar(agg_ftanet: dict, agg_scratch: dict, out: Path) -> None:
    x    = np.arange(len(LABELS))
    w    = 0.32
    vals_fta = [agg_ftanet[m] * 100 for m in METRICS]
    vals_sc  = [agg_scratch[m] * 100 for m in METRICS]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_fta = ax.bar(x - w/2, vals_fta, w, label="FTANet",
                      color=COLORS["FTANet"], alpha=0.85)
    bars_sc  = ax.bar(x + w/2, vals_sc,  w, label="SwiftF0-scratch",
                      color=COLORS["SwiftF0-scratch"], alpha=0.85)

    for bar, v in list(zip(bars_fta, vals_fta)) + list(zip(bars_sc, vals_sc)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x); ax.set_xticklabels(LABELS, fontsize=11)
    ax.set_ylabel("Score (%)")
    ax.set_title("FTANet vs SwiftF0-scratch — SCMS test set (original audio)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10)
    ax.axhline(90.27, color=COLORS["FTANet"],         ls=":", lw=1, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {out}")


def plot_scatter(per_stem_fta: list[dict], per_stem_sc: list[dict], out: Path) -> None:
    oa_fta = np.array([s["Overall Accuracy"] * 100 for s in per_stem_fta])
    oa_sc  = np.array([s["Overall Accuracy"] * 100 for s in per_stem_sc])
    diff   = oa_sc - oa_fta
    colors = np.where(diff > 0, COLORS["SwiftF0-scratch"], COLORS["FTANet"])

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(oa_fta, oa_sc, c=colors, s=18, alpha=0.6)
    lim = [min(oa_fta.min(), oa_sc.min()) - 2, 102]
    ax.plot(lim, lim, "k--", lw=0.8, alpha=0.5, label="parity")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("FTANet OA (%)", fontsize=11)
    ax.set_ylabel("SwiftF0-scratch OA (%)", fontsize=11)
    ax.set_title("Per-stem OA: FTANet vs SwiftF0-scratch")

    n_sc_wins  = (diff > 0).sum()
    n_fta_wins = (diff < 0).sum()
    handles = [
        mpatches.Patch(color=COLORS["SwiftF0-scratch"],
                       label=f"scratch wins ({n_sc_wins})"),
        mpatches.Patch(color=COLORS["FTANet"],
                       label=f"FTANet wins ({n_fta_wins})"),
    ]
    ax.legend(handles=handles, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {out}")


def plot_vfa_hist(per_stem_fta: list[dict], per_stem_sc: list[dict], out: Path) -> None:
    vfa_fta = np.array([s["Voicing False Alarm"] * 100 for s in per_stem_fta])
    vfa_sc  = np.array([s["Voicing False Alarm"] * 100 for s in per_stem_sc])

    bins = np.linspace(0, 60, 31)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(vfa_fta, bins=bins, alpha=0.65, color=COLORS["FTANet"],
            label=f"FTANet  (mean={vfa_fta.mean():.1f}%)")
    ax.hist(vfa_sc,  bins=bins, alpha=0.65, color=COLORS["SwiftF0-scratch"],
            label=f"SwiftF0-scratch  (mean={vfa_sc.mean():.1f}%)")
    ax.set_xlabel("Voicing False Alarm per stem (%)", fontsize=11)
    ax.set_ylabel("# stems")
    ax.set_title("VFA distribution — bleeding hypothesis")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {out}")


def plot_rpa_vs_vfa(per_stem_fta: list[dict], per_stem_sc: list[dict], out: Path) -> None:
    """RPA vs VFA scatter — quality vs false alarms trade-off."""
    for data, label, color in [
        (per_stem_fta, "FTANet",          COLORS["FTANet"]),
        (per_stem_sc,  "SwiftF0-scratch", COLORS["SwiftF0-scratch"]),
    ]:
        rpa = np.array([s["Raw Pitch Accuracy"]  * 100 for s in data])
        vfa = np.array([s["Voicing False Alarm"] * 100 for s in data])
        plt.figure(1)
        plt.scatter(vfa, rpa, s=14, alpha=0.5, color=color, label=label)

    fig = plt.figure(1)
    ax  = fig.axes[0]
    ax.set_xlabel("VFA (%)")
    ax.set_ylabel("RPA (%)")
    ax.set_title("RPA vs VFA per stem — pitch accuracy vs false alarms")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thr",         type=float, default=0.75)
    parser.add_argument("--run-scratch", default=None)
    parser.add_argument("--scms-root",   default=None)
    args = parser.parse_args()

    scms_root = Path(args.scms_root) if args.scms_root else SCMS_ROOT
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_tag   = args.run_scratch or "latest"
    thr_tag   = f"thr{args.thr:.2f}".replace(".", "")
    OUT_DIR   = OUT_BASE / f"{run_tag}_{thr_tag}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    _, test_stems = scms_official_split(scms_root)
    ann_dir = scms_root / "annotations" / "melody"
    test_stems = [s for s in test_stems if (ann_dir / f"{s}.csv").exists()]
    print(f"Test stems: {len(test_stems)}")

    # ── load scratch model ───────────────────────────────────────────────────
    print("Loading SwiftF0-scratch ...")
    model_sc = load_swiftf0("scratch", args.run_scratch, device)

    # ── collect per-stem predictions ─────────────────────────────────────────
    per_stem_fta, per_stem_sc = [], []
    n_skip_fta = 0

    for i, stem in enumerate(test_stems):
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(test_stems)}", end="\r", flush=True)

        ref_t, ref_f = load_annotation(ann_dir / f"{stem}.csv")

        # FTANet (cached)
        res = load_ftanet(stem, "original")
        if res is None:
            n_skip_fta += 1
            continue
        est_t_fta, est_f_fta = res

        # SwiftF0-scratch (live)
        wav = swiftf0_audio_path(stem, "original")
        if not wav.exists():
            continue
        t_sc, p_sc, c_sc = infer_swiftf0(model_sc, wav, device, SR_SC, HOP_SC)
        est_f_sc = apply_threshold(p_sc, c_sc, args.thr)

        sc_fta = eval_per_stem(ref_t, ref_f, est_t_fta, est_f_fta)
        sc_sc  = eval_per_stem(ref_t, ref_f, t_sc,      est_f_sc)
        if sc_fta and sc_sc:
            per_stem_fta.append(sc_fta)
            per_stem_sc.append(sc_sc)

    print(f"\n  evaluated {len(per_stem_fta)} stems  (FTANet missing: {n_skip_fta})")

    # ── aggregate ────────────────────────────────────────────────────────────
    def agg(per_stem):
        return {m: np.mean([s[m] for s in per_stem]) for m in METRICS}

    agg_fta = agg(per_stem_fta)
    agg_sc  = agg(per_stem_sc)

    print("\n── aggregate ─────────────────────────────────────────────")
    print(f"  {'Metric':<22} {'FTANet':>8}  {'Scratch':>8}")
    for m, lbl in zip(METRICS, LABELS):
        print(f"  {lbl:<22} {agg_fta[m]*100:>8.2f}  {agg_sc[m]*100:>8.2f}")

    # ── plots ─────────────────────────────────────────────────────────────────
    tag = f"{run_tag}_{thr_tag}"
    plot_bar(agg_fta, agg_sc,
             OUT_DIR / f"bar_metrics_{tag}.png")
    plot_scatter(per_stem_fta, per_stem_sc,
                 OUT_DIR / f"scatter_oa_{tag}.png")
    plot_vfa_hist(per_stem_fta, per_stem_sc,
                  OUT_DIR / f"hist_vfa_{tag}.png")
    plot_rpa_vs_vfa(per_stem_fta, per_stem_sc,
                    OUT_DIR / f"scatter_rpa_vfa_{tag}.png")


if __name__ == "__main__":
    main()
