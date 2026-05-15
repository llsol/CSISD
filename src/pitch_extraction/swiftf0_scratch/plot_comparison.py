"""
Plot SCMS sweep-evaluation results from a JSON produced by evaluate_scms.py.

Plots
-----
  tradeoff   VFA vs OA (or RPA) — one curve per system × source, thr as parameter
  sweep      all metrics vs threshold — one panel per system × source
  bar        grouped bar chart at --thr or best-OA threshold per system

Usage
-----
  python -m src.pitch_extraction.swiftf0_scratch.plot_comparison \\
      data/results/scms_eval/scms_eval_2026-05-15_132059_sweep.json

  python -m src.pitch_extraction.swiftf0_scratch.plot_comparison \\
      data/results/scms_eval/... --plot tradeoff --y-metric RPA --sources as

  python -m src.pitch_extraction.swiftf0_scratch.plot_comparison \\
      data/results/scms_eval/... --plot bar --thr 0.85 --sources original as

  python -m src.pitch_extraction.swiftf0_scratch.plot_comparison \\
      data/results/scms_eval/... --plot all --show
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import settings

OUT_BASE = settings.FIGURES_DIR / "swiftf0_scratch" / "evaluation"

# long metric name → short label
SHORT = {
    "Voicing Recall":     "VR",
    "Voicing False Alarm":"VFA",
    "Raw Pitch Accuracy": "RPA",
    "Raw Chroma Accuracy":"RCA",
    "Overall Accuracy":   "OA",
}
LONG = {v: k for k, v in SHORT.items()}   # short → long

SYS_COLOR = {
    "FTANet":           "#1f77b4",
    "SwiftF0-finetune": "#ff7f0e",
    "SwiftF0-scratch":  "#d62728",
}
SRC_LS  = {"original": "-",  "as": "--"}
SRC_MRK = {"original": "o",  "as": "s"}
MET_COLOR = {
    "VR": "#2ca02c", "VFA": "#d62728", "RPA": "#1f77b4",
    "RCA": "#9467bd", "OA": "#ff7f0e",
}


# ── data helpers ──────────────────────────────────────────────────────────────

def load_json(path: Path):
    data = json.loads(path.read_text())
    return data, data["results"]


def _sweep_pts(results, system, source):
    return sorted(
        [e for e in results
         if e["system"] == system and e["source"] == source
         and e.get("best") is False],
        key=lambda e: e["threshold"],
    )


def _best_entry(results, system, source):
    pts = [e for e in results
           if e["system"] == system and e["source"] == source
           and (e.get("best") is True or e.get("best") is None)]
    return pts[0] if pts else None


def _at_thr(results, system, source, thr):
    pts = _sweep_pts(results, system, source)
    if not pts:
        return _best_entry(results, system, source)
    return min(pts, key=lambda e: abs(e["threshold"] - thr))


def _save_or_show(fig, out: Path | None, show: bool):
    if show:
        matplotlib.use("TkAgg")
        plt.show()
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"→ {out}")
    plt.close(fig)


# ── plot: tradeoff ────────────────────────────────────────────────────────────

def plot_tradeoff(meta, results, systems, sources, y_short, out, show):
    y_long = LONG[y_short]
    baselines = meta.get("paper_baselines", {})

    fig, ax = plt.subplots(figsize=(10, 6))

    for system in systems:
        color = SYS_COLOR.get(system, "gray")
        for source in sources:
            ls  = SRC_LS.get(source, "-")
            mrk = SRC_MRK.get(source, "o")
            pts = _sweep_pts(results, system, source)

            if not pts:
                # FTANet: single diamond
                e = _best_entry(results, system, source)
                if e:
                    ax.scatter(e["Voicing False Alarm"], e[y_long],
                               color=color, marker="D", s=110, zorder=6,
                               label=f"{system} / {source}")
                continue

            vfa = [e["Voicing False Alarm"] for e in pts]
            ym  = [e[y_long]                for e in pts]
            thr = [e["threshold"]           for e in pts]

            ax.plot(vfa, ym, color=color, ls=ls, lw=1.8,
                    label=f"{system} / {source}")
            ax.scatter(vfa, ym, color=color, marker=mrk, s=28, zorder=4)

            # label first, mid, last threshold
            for i in [0, len(thr) // 2, -1]:
                ax.annotate(f"{thr[i]:.2f}", (vfa[i], ym[i]),
                            textcoords="offset points", xytext=(0, 5),
                            fontsize=6, ha="center", color=color)

            # star on best-OA point
            best = _best_entry(results, system, source)
            if best and best.get("threshold"):
                ax.scatter(best["Voicing False Alarm"], best[y_long],
                           color=color, marker="*", s=250, zorder=7,
                           edgecolors="k", linewidths=0.5)

    # paper baselines
    for name, b in baselines.items():
        if y_short not in b:
            continue
        ax.scatter(b["VFA"], b[y_short], marker="x", s=90,
                   color="black", zorder=8, linewidths=1.5)
        ax.annotate(name, (b["VFA"], b[y_short]),
                    textcoords="offset points", xytext=(5, -9),
                    fontsize=7, color="black")

    ax.set_xlabel("Voicing False Alarm (%)", fontsize=12)
    ax.set_ylabel(f"{y_short} (%)", fontsize=12)
    ax.set_title(
        f"Trade-off: VFA vs {y_short}\n"
        f"★ = best-OA threshold   labels = threshold value   ◆ = FTANet"
    )
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_or_show(fig, out, show)


# ── plot: sweep ───────────────────────────────────────────────────────────────

def plot_sweep(meta, results, systems, sources, out, show):
    sweep_sys = [s for s in systems if s != "FTANet"]
    panels    = [(sys, src) for sys in sweep_sys for src in sources
                 if _sweep_pts(results, sys, src)]
    if not panels:
        print("[sweep] no sweep entries found — skipping")
        return

    ncols = len(sources)
    nrows = len(sweep_sys)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6 * ncols, 4 * nrows),
                             squeeze=False)

    for ri, system in enumerate(sweep_sys):
        for ci, source in enumerate(sources):
            ax  = axes[ri][ci]
            pts = _sweep_pts(results, system, source)
            if not pts:
                ax.set_visible(False)
                continue

            thr_vals = [e["threshold"] for e in pts]
            for long_name, short_name in SHORT.items():
                if long_name not in pts[0]:
                    continue
                vals = [e[long_name] for e in pts]
                ax.plot(thr_vals, vals, color=MET_COLOR[short_name],
                        lw=1.6, label=short_name)

            # vertical dashed line at best threshold
            best = _best_entry(results, system, source)
            if best and best.get("threshold"):
                ax.axvline(best["threshold"], color="black", ls=":", lw=1.2,
                           label=f"best thr={best['threshold']:.2f}")

            ax.set_title(f"{system} / {source}", fontsize=10)
            ax.set_xlabel("Threshold")
            ax.set_ylabel("Score (%)")
            ax.legend(fontsize=7, ncol=3)
            ax.grid(True, alpha=0.3)

    fig.suptitle("Metrics vs threshold", fontsize=13)
    fig.tight_layout()
    _save_or_show(fig, out, show)


# ── plot: bar ─────────────────────────────────────────────────────────────────

def plot_bar(meta, results, systems, sources, thr, out, show):
    baselines = meta.get("paper_baselines", {})
    keys   = [(sys, src) for sys in systems for src in sources]
    n_bars = len(keys)
    n_grp  = len(SHORT)
    w      = 0.75 / max(n_bars, 1)
    x      = np.arange(n_grp)

    fig, ax = plt.subplots(figsize=(12, 5))

    for i, (system, source) in enumerate(keys):
        e = (_at_thr(results, system, source, thr) if thr is not None
             else _best_entry(results, system, source))
        if e is None:
            continue

        vals  = [e[long] for long in SHORT]
        color = SYS_COLOR.get(system, "gray")
        t_used = e.get("threshold")
        t_str  = f" thr={t_used:.2f}" if t_used is not None else ""
        src_str = "orig" if source == "original" else "as"
        label   = f"{system}/{src_str}{t_str}"

        offset = i * w - (n_bars - 1) * w / 2
        bars   = ax.bar(x + offset, vals, w, label=label,
                        color=color,
                        alpha=0.85 if source == "original" else 0.55,
                        edgecolor="k" if source == "as" else "none",
                        linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=6)

    # paper baselines — OA as reference line
    for name, b in baselines.items():
        oa_idx = list(SHORT.keys()).index("Overall Accuracy")
        ax.plot([x[oa_idx] - 0.4, x[oa_idx] + 0.4],
                [b["OA"], b["OA"]], color="black", ls=":", lw=1.2)
        ax.annotate(f"{name} OA={b['OA']:.1f}",
                    (x[oa_idx] + 0.4, b["OA"]),
                    fontsize=6, va="center")

    thr_title = f"thr={thr:.2f}" if thr is not None else "best-OA threshold per system"
    ax.set_xticks(x)
    ax.set_xticklabels(list(SHORT.values()), fontsize=11)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 107)
    ax.set_title(f"Metrics comparison — {thr_title}")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    _save_or_show(fig, out, show)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot SCMS sweep evaluation results from JSON."
    )
    parser.add_argument("results", type=Path,
                        help="Path to scms_eval_*.json")
    parser.add_argument("--plot", default="all",
                        choices=["tradeoff", "sweep", "bar", "all"],
                        help="Which plot(s) to produce")
    parser.add_argument("--sources", nargs="+", default=None,
                        choices=["original", "as"],
                        help="Filter sources (default: all in JSON)")
    parser.add_argument("--systems", nargs="+", default=None,
                        help="Filter systems (default: all in JSON)")
    parser.add_argument("--y-metric", default="OA",
                        choices=list(LONG.keys()),
                        help="Y-axis metric for tradeoff plot (default: OA)")
    parser.add_argument("--thr", type=float, default=None,
                        help="Fixed threshold for bar plot (default: best per system)")
    parser.add_argument("--show", action="store_true",
                        help="Display plots interactively instead of saving")
    args = parser.parse_args()

    meta, results = load_json(args.results)

    all_systems = list(dict.fromkeys(e["system"] for e in results))
    all_sources = list(dict.fromkeys(e["source"] for e in results))

    systems = args.systems or all_systems
    sources = args.sources or all_sources

    tag     = args.results.stem
    out_dir = OUT_BASE / tag
    do      = args.plot

    if do in ("tradeoff", "all"):
        plot_tradeoff(
            meta, results, systems, sources, args.y_metric,
            out_dir / f"tradeoff_vfa_{args.y_metric}_{tag}.png", args.show,
        )
    if do in ("sweep", "all"):
        plot_sweep(
            meta, results, systems, sources,
            out_dir / f"sweep_{tag}.png", args.show,
        )
    if do in ("bar", "all"):
        thr_tag = f"thr{args.thr:.2f}".replace(".", "") if args.thr is not None else "best"
        plot_bar(
            meta, results, systems, sources, args.thr,
            out_dir / f"bar_{thr_tag}_{tag}.png", args.show,
        )


if __name__ == "__main__":
    main()
