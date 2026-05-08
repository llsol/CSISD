"""
Compare generated svaras (GRU+VAE prior samples) vs ground-truth svaras
in feature space and segment succession.

Feature comparison (violin plots):
    cp_frac, sta_frac, sil_frac, tr_frac  — time fraction per segment type
    n_cp, n_sta                            — segment counts
    cp_mean_cents, sta_mean_cents          — pitch position

Succession / transition analysis:
    4×4 transition probability matrices (GT vs gen, pooled + per svara)
    First / last segment distributions
    Sequence length histograms
    Top-10 most common sequences
    Hard-rule violation check

Output: figures/gruvae/{run}/compare_generated{_hardmask}/
    overview.png                    — feature violins 4×2 grid
    violins/violin_{feat}.png       — individual feature violins
    numeric_summary.txt / .csv      — mean/std/Δ/Cohen's d per feature × svara
    proposals.txt                   — lambda & hard-rule adjustment suggestions
    transition_pooled.png           — pooled 4×4 heatmaps + first/last/length
    transition_per_svara.png        — per-svara heatmaps
    transition_summary.txt          — numeric tables + top seqs + rule violations

Usage:
    python -m src.models.gruvae.compare_generated
    python -m src.models.gruvae.compare_generated --run run_001 --n 300
    python -m src.models.gruvae.compare_generated --hard-mask
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import settings
from src.models.gruvae.dataset_gruvae import SVARA_LABELS, build_corpus_sequences
from src.models.gruvae.generate_plot import load_model, generate
from src.models.gruvae.model_gruvae import ModelConfig
from src.analysis.svara_segment_analysis import load_all, SCALE_ORDER

import torch

CKPT_DIR = settings.DATA_INTERIM / "models" / "gruvae_v4"

SCALE_ORDER_PRESENT = [s for s in SCALE_ORDER if s in set(SVARA_LABELS)]

COLOR_GT  = "#4477AA"
COLOR_GEN = "#EE6677"

# ── segment type ordering ─────────────────────────────────────────────────────

TYPE_ORD = ["CP", "SIL", "STA", "TR"]
TYPE_IDX = {t: i for i, t in enumerate(TYPE_ORD)}

VALID_NEXT: dict[str, set[str]] = {
    "CP":  {"CP", "SIL", "STA", "TR"},
    "SIL": {"CP", "STA", "TR"},
    "STA": {"SIL", "STA", "TR"},
    "TR":  {"CP"},
}

# ── feature definitions ───────────────────────────────────────────────────────

FEATURES = [
    ("cp_frac",        "CP fraction"),
    ("sta_frac",       "STA fraction"),
    ("sil_frac",       "SIL fraction"),
    ("tr_frac",        "TR fraction"),
    ("n_cp",           "n CP segments"),
    ("n_sta",          "n STA segments"),
    ("cp_mean_cents",  "CP mean (cents)"),
    ("sta_mean_cents", "STA mean (cents)"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Data extraction
# ═══════════════════════════════════════════════════════════════════════════════

def _feats_from_sample(sv_data: dict) -> dict:
    segs      = sv_data["segments"]
    total_rel = sum(s["dur_rel"] for s in segs) or 1.0

    cp_segs  = [s for s in segs if s["type"] == "CP"]
    sta_segs = [s for s in segs if s["type"] == "STA"]
    sil_segs = [s for s in segs if s["type"] == "SIL"]
    tr_segs  = [s for s in segs if s["type"] == "TR"]

    def _frac(sub):
        return sum(s["dur_rel"] for s in sub) / total_rel

    cp_cents  = [s["cents"] for s in cp_segs  if s["cents"] != 0.0]
    sta_cents = [s["cents"] for s in sta_segs if s["cents"] != 0.0]

    return {
        "svara_label":    sv_data["svara"],
        "cp_frac":        _frac(cp_segs),
        "sta_frac":       _frac(sta_segs),
        "sil_frac":       _frac(sil_segs),
        "tr_frac":        _frac(tr_segs),
        "n_cp":           float(len(cp_segs)),
        "n_sta":          float(len(sta_segs)),
        "cp_mean_cents":  float(np.mean(cp_cents))  if cp_cents  else np.nan,
        "sta_mean_cents": float(np.mean(sta_cents)) if sta_cents else np.nan,
        "_type_seq":      [s["type"] for s in segs],
    }


def load_gt_data(svaras: list[str]) -> tuple[list[dict], dict[str, list[list[str]]]]:
    """
    Returns (gt_rows, gt_seqs_by_svara).
    gt_rows: feature dicts (scalar, same keys as _feats_from_sample minus _type_seq)
    gt_seqs_by_svara: {svara: [[type_seq], ...]}
    """
    print("[compare_gen] loading GT svaras (feature stats + type sequences)...")
    all_rows, _, svara_labels, _ = load_all()
    gt_rows = [{k: v for k, v in r.items() if not isinstance(v, list)} for r in all_rows]

    # build type sequences from dataset builder (re-uses same segment logic)
    print("[compare_gen] building GT type sequences via dataset builder...")
    corpus_seqs = build_corpus_sequences()   # list of {svara_label, sequence: (n,7)}
    gt_seqs: dict[str, list[list[str]]] = {sv: [] for sv in svaras}
    for item in corpus_seqs:
        sv = item["svara_label"]
        if sv in gt_seqs:
            seq = item["sequence"]          # (n_segs, 7): first 4 cols are one-hot type
            types = [TYPE_ORD[int(np.argmax(row[:4]))] for row in seq]
            gt_seqs[sv].append(types)

    return gt_rows, gt_seqs


# ═══════════════════════════════════════════════════════════════════════════════
# Numeric feature analysis
# ═══════════════════════════════════════════════════════════════════════════════

def _clean(vals):
    a = np.array(vals, dtype=float)
    return a[np.isfinite(a)]


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else np.nan


def _effect_tag(d: float) -> str:
    if not np.isfinite(d): return "n/a"
    ad = abs(d)
    if ad < 0.2: return "negligible"
    if ad < 0.5: return "small"
    if ad < 0.8: return "medium"
    return "LARGE"


def compute_summary(gt_rows, gen_rows, svaras):
    summary = {}
    for feat_key, _ in FEATURES:
        per_sv = {}
        all_gt_v, all_gen_v = [], []
        for sv in svaras:
            gt_v  = _clean([r[feat_key] for r in gt_rows  if r.get("svara_label") == sv])
            gen_v = _clean([r[feat_key] for r in gen_rows if r.get("svara_label") == sv])
            all_gt_v.extend(gt_v); all_gen_v.extend(gen_v)
            per_sv[sv] = {
                "gt_mean":  float(np.mean(gt_v))    if len(gt_v)  else np.nan,
                "gt_std":   float(np.std(gt_v))     if len(gt_v)  else np.nan,
                "gt_med":   float(np.median(gt_v))  if len(gt_v)  else np.nan,
                "gen_mean": float(np.mean(gen_v))   if len(gen_v) else np.nan,
                "gen_std":  float(np.std(gen_v))    if len(gen_v) else np.nan,
                "gen_med":  float(np.median(gen_v)) if len(gen_v) else np.nan,
                "delta":    float(np.mean(gen_v) - np.mean(gt_v)) if (len(gt_v) and len(gen_v)) else np.nan,
                "cohens_d": cohens_d(gen_v, gt_v),
            }
        ag, agg = _clean(all_gt_v), _clean(all_gen_v)
        summary[feat_key] = {
            "pooled": {
                "gt_mean":  float(np.mean(ag))    if len(ag)  else np.nan,
                "gt_std":   float(np.std(ag))     if len(ag)  else np.nan,
                "gt_med":   float(np.median(ag))  if len(ag)  else np.nan,
                "gen_mean": float(np.mean(agg))   if len(agg) else np.nan,
                "gen_std":  float(np.std(agg))    if len(agg) else np.nan,
                "gen_med":  float(np.median(agg)) if len(agg) else np.nan,
                "delta":    float(np.mean(agg) - np.mean(ag)) if (len(ag) and len(agg)) else np.nan,
                "cohens_d": cohens_d(agg, ag),
            },
            "per_svara": per_sv,
        }
    return summary


def print_and_save_summary(summary, svaras, out_dir, run_name, n_gen):
    lines = [f"Numeric summary — GRU+VAE [{run_name}]  n_gen={n_gen}", "=" * 90]
    csv_rows = []
    for feat_key, feat_label in FEATURES:
        p   = summary[feat_key]["pooled"]
        d   = p["cohens_d"]
        tag = _effect_tag(d)
        lines.append(f"\n{feat_label}  (pooled Cohen's d={d:+.3f}  [{tag}])")
        lines.append(f"  {'':12s}  {'GT mean':>9}  {'GT std':>8}  {'Gen mean':>9}  {'Gen std':>8}  {'Δ':>8}")
        lines.append(f"  {'[pooled]':12s}  {p['gt_mean']:>9.3f}  {p['gt_std']:>8.3f}  "
                     f"{p['gen_mean']:>9.3f}  {p['gen_std']:>8.3f}  {p['delta']:>+8.3f}")
        for sv in svaras:
            s    = summary[feat_key]["per_svara"][sv]
            d_sv = s["cohens_d"]
            d_str = f"  d={d_sv:+.2f}" if np.isfinite(d_sv) else "  d=n/a"
            lines.append(f"  {sv:12s}  {s['gt_mean']:>9.3f}  {s['gt_std']:>8.3f}  "
                         f"{s['gen_mean']:>9.3f}  {s['gen_std']:>8.3f}  {s['delta']:>+8.3f}{d_str}")
            csv_rows.append({
                "feature": feat_key, "svara": sv,
                "gt_mean": s["gt_mean"], "gt_std": s["gt_std"], "gt_med": s["gt_med"],
                "gen_mean": s["gen_mean"], "gen_std": s["gen_std"], "gen_med": s["gen_med"],
                "delta": s["delta"], "cohens_d": s["cohens_d"],
            })
    text = "\n".join(lines)
    print(text)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "numeric_summary.txt").write_text(text)
    print(f"\n[compare_gen] → {out_dir / 'numeric_summary.txt'}")
    with open(out_dir / "numeric_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader(); w.writerows(csv_rows)
    print(f"[compare_gen] → {out_dir / 'numeric_summary.csv'}")


# ═══════════════════════════════════════════════════════════════════════════════
# Lambda & hard-rule proposals
# ═══════════════════════════════════════════════════════════════════════════════

def _scale_factor(d_abs):
    if d_abs < 0.2: return 1.0
    if d_abs < 0.5: return 1.35
    if d_abs < 0.8: return 1.7
    return 2.5


def propose_adjustments(summary, cfg, svaras, out_dir):
    lines = ["Lambda & hard-rule adjustment proposals", "=" * 70,
             "Legend: Δ = gen_mean − gt_mean  |d| = pooled Cohen's d",
             "        ↑ increase lambda   ↓ decrease lambda   ≈ no change needed", ""]
    lambda_changes = {}

    lines.append("── Duration fractions ────────────────────────────────────────")
    for feat_key, lam_name, cur in [
        ("cp_frac",  "lambda_dur_cp",  getattr(cfg, "lambda_dur_cp",  0.3)),
        ("sta_frac", "lambda_dur_sta", getattr(cfg, "lambda_dur_sta", 0.05)),
        ("sil_frac", "lambda_dur_sil", getattr(cfg, "lambda_dur_sil", 0.1)),
        ("tr_frac",  "lambda_dur_tr",  getattr(cfg, "lambda_dur_tr",  0.1)),
    ]:
        p   = summary[feat_key]["pooled"]
        d   = p["cohens_d"]
        delta = p["delta"]
        tag = _effect_tag(d)
        if not np.isfinite(d) or abs(d) < 0.2:
            lines.append(f"  {feat_key:14s}  Δ={delta:+.3f}  |d|={abs(d) if np.isfinite(d) else float('nan'):.2f}  [{tag}]  → no change")
            continue
        sc  = _scale_factor(abs(d))
        nv  = round(cur * sc if delta > 0 else cur / sc, 4)
        arrow = "↑" if delta > 0 else "↓"
        reason = "over-generated" if delta > 0 else "under-generated"
        lambda_changes[lam_name] = nv
        lines.append(f"  {feat_key:14s}  Δ={delta:+.3f}  |d|={abs(d):.2f}  [{tag}]  "
                     f"{arrow} {lam_name} {cur} → {nv}  (gen {reason})")

    lines += ["", "── Pitch (cents) ──────────────────────────────────────────────"]
    for feat_key, lam_name, cur in [
        ("cp_mean_cents",  "lambda_cp_cents",  getattr(cfg, "lambda_cp_cents",  2.0)),
        ("sta_mean_cents", "lambda_sta_cents", getattr(cfg, "lambda_sta_cents", 5.0)),
    ]:
        p     = summary[feat_key]["pooled"]
        d, delta = p["cohens_d"], p["delta"]
        tag   = _effect_tag(d)
        d_disp = f"{abs(d):.2f}" if np.isfinite(d) else "nan"
        if not np.isfinite(d) or abs(d) < 0.2:
            lines.append(f"  {feat_key:14s}  Δ={delta:+.1f}¢  |d|={d_disp}  [{tag}]  → no change")
            continue
        nv = round(cur * _scale_factor(abs(d)), 2)
        lambda_changes[lam_name] = nv
        lines.append(f"  {feat_key:14s}  Δ={delta:+.1f}¢  |d|={abs(d):.2f}  [{tag}]  "
                     f"↑ {lam_name} {cur} → {nv}  (pitch bias — increase reconstruction weight)")

    lines += ["", "── Segment counts & hard-rule suggestions ─────────────────────"]
    for feat_key in ("n_cp", "n_sta"):
        p     = summary[feat_key]["pooled"]
        d, delta = p["cohens_d"], p["delta"]
        tag   = _effect_tag(d)
        d_disp = f"{abs(d):.2f}" if np.isfinite(d) else "nan"
        lines.append(f"  {feat_key:14s}  Δ={delta:+.2f}  |d|={d_disp}  [{tag}]")
        if not np.isfinite(d) or abs(d) < 0.2:
            lines.append("               → no change"); continue
        if feat_key == "n_cp":
            cp_frac_delta = summary["cp_frac"]["pooled"]["delta"]
            if delta > 0 and abs(cp_frac_delta) < 0.05:
                lines += ["               → more CP *segments* but similar cp_frac → fragmentation",
                          "                 Hard-rule: restrict CP→CP  VALID_NEXT['CP'] -= {'CP'}",
                          "                 (trade-off: long stable regions are musically valid)"]
            elif delta > 0:
                lines += ["               → model over-generates CP overall",
                          "                 Primary fix: increase lambda_dur_cp (see above)"]
            else:
                lines += ["               → model under-generates CP segments",
                          "                 Check: CP→SIL→CP suppressed? Try reducing lambda_dur_sil"]
        else:  # n_sta
            sta_delta = summary["sta_frac"]["pooled"]["delta"]
            if delta < 0:
                lines += ["               → model under-generates STA segments",
                          "                 Hard-rule options (pick one):",
                          "                   1. Allow STA→CP: VALID_NEXT['STA'].add('CP')",
                          "                      Risk: skips return glide (TR) — less musical",
                          "                   2. Keep rules; reduce lambda_dur_sta if also sta_frac low"]
                if abs(sta_delta) > 0.05:
                    lines.append("                   3. Reduce lambda_dur_sta (see above)")
            else:
                lines += ["               → model over-generates STA segments",
                          "                 Hard-rule: restrict STA→STA: VALID_NEXT['STA'] -= {'STA'}",
                          "                   (force SIL or TR between consecutive STAs)"]

    if lambda_changes:
        lines += ["", "── Suggested ModelConfig changes ──────────────────────────────"]
        for k, v in sorted(lambda_changes.items()):
            cur = getattr(cfg, k, "?")
            lines.append(f"  {k} = {v}   # was {cur}")
    else:
        lines += ["", "  No significant lambda changes needed (all |d| < 0.2)."]

    lines += ["", "Note: proposals are data-driven but not causal.",
              "      Verify musicologically before changing hard rules."]

    text = "\n".join(lines)
    print("\n" + text)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "proposals.txt").write_text(text)
    print(f"\n[compare_gen] → {out_dir / 'proposals.txt'}")


# ═══════════════════════════════════════════════════════════════════════════════
# Transition / succession analysis
# ═══════════════════════════════════════════════════════════════════════════════

def _transition_matrix(sequences: list[list[str]]) -> np.ndarray:
    mat = np.zeros((4, 4), dtype=float)
    for seq in sequences:
        for a, b in zip(seq[:-1], seq[1:]):
            if a in TYPE_IDX and b in TYPE_IDX:
                mat[TYPE_IDX[a], TYPE_IDX[b]] += 1
    return mat


def _row_normalize(mat: np.ndarray) -> np.ndarray:
    rs = mat.sum(axis=1, keepdims=True)
    return np.where(rs > 0, mat / rs, 0.0)


def _pool(seqs_by_svara: dict, svaras: list[str]) -> list[list[str]]:
    out = []
    for sv in svaras:
        out.extend(seqs_by_svara.get(sv, []))
    return out


def _heatmap(ax, mat, title, cmap="Blues", vmin=0, vmax=1, annot_color_thr=0.5):
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(4)); ax.set_xticklabels(TYPE_ORD, fontsize=8)
    ax.set_yticks(range(4)); ax.set_yticklabels(TYPE_ORD, fontsize=8)
    ax.set_xlabel("Next", fontsize=7); ax.set_ylabel("Current", fontsize=7)
    ax.set_title(title, fontsize=9)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(mat[i,j]) > annot_color_thr else "black")
    return im


def _first_last_dist(sequences: list[list[str]], which: str) -> list[float]:
    counts = {t: 0 for t in TYPE_ORD}
    for seq in sequences:
        if seq:
            t = seq[0] if which == "first" else seq[-1]
            counts[t] = counts.get(t, 0) + 1
    total = sum(counts.values()) or 1
    return [counts[t] / total for t in TYPE_ORD]


def plot_transition_analysis(
    gt_seqs:  dict[str, list[list[str]]],
    gen_seqs: dict[str, list[list[str]]],
    svaras:   list[str],
    out_dir:  Path,
    run_name: str,
    n_gen:    int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_gt  = _pool(gt_seqs,  svaras)
    all_gen = _pool(gen_seqs, svaras)

    gt_mat  = _row_normalize(_transition_matrix(all_gt))
    gen_mat = _row_normalize(_transition_matrix(all_gen))
    diff_mat = gen_mat - gt_mat

    # ── pooled figure (2 rows × 3 cols) ──────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(
        f"Segment succession — GRU+VAE [{run_name}]  n_gen={n_gen}/svara",
        fontsize=12, fontweight="bold",
    )

    _heatmap(axes[0, 0], gt_mat,   "GT — P(next|current)")
    _heatmap(axes[0, 1], gen_mat,  "Generated — P(next|current)")
    _heatmap(axes[0, 2], diff_mat, "Δ = Generated − GT",
             cmap="RdBu_r", vmin=-1, vmax=1, annot_color_thr=0.4)

    x = np.arange(4); w = 0.35
    for col, which in [(0, "first"), (1, "last")]:
        ax = axes[1, col]
        ax.bar(x - w/2, _first_last_dist(all_gt,  which), w,
               label="GT",  color=COLOR_GT,  alpha=0.75)
        ax.bar(x + w/2, _first_last_dist(all_gen, which), w,
               label="Gen", color=COLOR_GEN, alpha=0.75)
        ax.set_xticks(x); ax.set_xticklabels(TYPE_ORD)
        ax.set_ylabel("Fraction"); ax.legend(fontsize=8)
        ax.set_title(f"{which.capitalize()} segment type")
        ax.grid(axis="y", lw=0.4, alpha=0.4)

    ax = axes[1, 2]
    gt_lens  = [len(s) for s in all_gt]
    gen_lens = [len(s) for s in all_gen]
    max_len  = max(max(gt_lens, default=1), max(gen_lens, default=1))
    bins = np.arange(0.5, max_len + 1.5, 1)
    ax.hist(gt_lens,  bins=bins, density=True, alpha=0.65,
            color=COLOR_GT,  label=f"GT (μ={np.mean(gt_lens):.1f})")
    ax.hist(gen_lens, bins=bins, density=True, alpha=0.65,
            color=COLOR_GEN, label=f"Gen (μ={np.mean(gen_lens):.1f})")
    ax.set_xlabel("n segments"); ax.set_ylabel("Density")
    ax.set_title("Sequence length distribution"); ax.legend(fontsize=8)
    ax.grid(axis="y", lw=0.4, alpha=0.4)

    plt.tight_layout()
    p1 = out_dir / "transition_pooled.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare_gen] → {p1}")

    # ── per-svara heatmaps ────────────────────────────────────────────────────
    n_sv = len(svaras)
    fig, axes = plt.subplots(n_sv, 3, figsize=(11, max(4, n_sv * 3.0)), squeeze=False)
    fig.suptitle(f"Transitions per svara [{run_name}]", fontsize=11, fontweight="bold")

    for r, sv in enumerate(svaras):
        gt_sv  = gt_seqs.get(sv, [])
        gen_sv = gen_seqs.get(sv, [])
        gt_m   = _row_normalize(_transition_matrix(gt_sv))
        gen_m  = _row_normalize(_transition_matrix(gen_sv))
        diff_m = gen_m - gt_m
        for c, (mat, title, cmap, vmin, vmax, thr) in enumerate([
            (gt_m,   f"{sv}  GT",    "Blues",   0,  1,    0.5),
            (gen_m,  f"{sv}  Gen",   "Blues",   0,  1,    0.5),
            (diff_m, f"{sv}  Δ",     "RdBu_r", -1,  1,    0.4),
        ]):
            ax = axes[r][c]
            ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_xticks(range(4)); ax.set_xticklabels(TYPE_ORD, fontsize=7)
            ax.set_yticks(range(4)); ax.set_yticklabels(TYPE_ORD, fontsize=7)
            ax.set_title(title, fontsize=8)
            for i in range(4):
                for j in range(4):
                    ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                            fontsize=6, color="white" if abs(mat[i,j]) > thr else "black")

    plt.tight_layout()
    p2 = out_dir / "transition_per_svara.png"
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare_gen] → {p2}")

    # ── text summary ──────────────────────────────────────────────────────────
    _transition_text_summary(all_gt, all_gen, gt_mat, gen_mat, gt_seqs, gen_seqs, svaras, out_dir)


def _transition_text_summary(
    all_gt, all_gen, gt_mat, gen_mat, gt_seqs, gen_seqs, svaras, out_dir
):
    lines = ["Transition / succession analysis", "=" * 65]

    def _mat_block(mat, label):
        out = [f"\n{label}"]
        out.append("  from \\ to " + "".join(f"  {t:4s}" for t in TYPE_ORD))
        for i, ft in enumerate(TYPE_ORD):
            out.append(f"  {ft:8s}  " + "".join(f"  {mat[i,j]:.2f}" for j in range(4)))
        return out

    lines += _mat_block(gt_mat,  "Pooled GT transition matrix (row = current, col = next):")
    lines += _mat_block(gen_mat, "Pooled Generated transition matrix:")
    lines += _mat_block(gen_mat - gt_mat, "Delta (Generated − GT):")

    diff = gen_mat - gt_mat
    lines.append("\nLargest deltas |Δ| > 0.05:")
    diffs = [(abs(diff[i,j]), diff[i,j], TYPE_ORD[i], TYPE_ORD[j])
             for i in range(4) for j in range(4) if abs(diff[i,j]) > 0.05]
    diffs.sort(reverse=True)
    for _, d, ft, tt in diffs:
        arrow = "↑ gen over-uses" if d > 0 else "↓ gen under-uses"
        lines.append(f"  {ft}→{tt}:  Δ={d:+.3f}  {arrow}")
    if not diffs:
        lines.append("  None.")

    lines.append("\nTop-10 GT sequences (pooled):")
    for seq, cnt in Counter("→".join(s) for s in all_gt).most_common(10):
        pct = cnt / len(all_gt) * 100
        lines.append(f"  {cnt:5d}  ({pct:5.1f}%)  {seq}")

    lines.append("\nTop-10 Generated sequences (pooled):")
    for seq, cnt in Counter("→".join(s) for s in all_gen).most_common(10):
        pct = cnt / len(all_gen) * 100
        lines.append(f"  {cnt:5d}  ({pct:5.1f}%)  {seq}")

    lines.append("\nSequences only in GT (top-5):")
    gt_set  = Counter("→".join(s) for s in all_gt)
    gen_set = Counter("→".join(s) for s in all_gen)
    only_gt = sorted([(c, s) for s, c in gt_set.items() if s not in gen_set], reverse=True)[:5]
    for cnt, seq in only_gt:
        lines.append(f"  {cnt:5d}  {seq}")
    if not only_gt:
        lines.append("  None (all GT sequences appear in generated too).")

    lines.append("\nHard-rule violations in Generated:")
    n_viol = 0
    viol_counts: Counter = Counter()
    for seq in all_gen:
        for a, b in zip(seq[:-1], seq[1:]):
            if b not in VALID_NEXT.get(a, set()):
                n_viol += 1
                viol_counts[f"{a}→{b}"] += 1
    if n_viol == 0:
        lines.append("  None — model respects all VALID_NEXT rules.")
    else:
        lines.append(f"  Total violations: {n_viol}")
        for pair, cnt in viol_counts.most_common():
            lines.append(f"    {pair}: {cnt}×")

    lines.append("\nHard-rule violations in GT (sanity check):")
    n_gt_viol = 0
    gt_viol_counts: Counter = Counter()
    for seq in all_gt:
        for a, b in zip(seq[:-1], seq[1:]):
            if b not in VALID_NEXT.get(a, set()):
                n_gt_viol += 1
                gt_viol_counts[f"{a}→{b}"] += 1
    if n_gt_viol == 0:
        lines.append("  None.")
    else:
        lines.append(f"  Total: {n_gt_viol}")
        for pair, cnt in gt_viol_counts.most_common():
            lines.append(f"    {pair}: {cnt}×  ← rule may be over-restrictive")

    text = "\n".join(lines)
    print("\n" + text)
    p = out_dir / "transition_summary.txt"
    p.write_text(text)
    print(f"\n[compare_gen] → {p}")


# ═══════════════════════════════════════════════════════════════════════════════
# Violin plots
# ═══════════════════════════════════════════════════════════════════════════════

def _violin_feature(ax, feat_key, feat_label, gt_rows, gen_rows, svaras, summary=None):
    n = len(svaras)
    pos_gt  = np.arange(n) * 3.0
    pos_gen = pos_gt + 1.0

    def _vals(rows, sv):
        return _clean([r[feat_key] for r in rows if r.get("svara_label") == sv]).tolist()

    for i, sv in enumerate(svaras):
        for vals, pos, color in (
            (_vals(gt_rows,  sv), pos_gt[i],  COLOR_GT),
            (_vals(gen_rows, sv), pos_gen[i], COLOR_GEN),
        ):
            if len(vals) < 2:
                ax.scatter([pos], vals or [np.nan], color=color, s=18, zorder=3)
                continue
            parts = ax.violinplot(vals, positions=[pos], widths=0.8,
                                  showmedians=True, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(color); pc.set_alpha(0.55)
            parts["cmedians"].set_color(color); parts["cmedians"].set_linewidth(1.5)

    ax.set_xticks((pos_gt + pos_gen) / 2)
    ax.set_xticklabels(svaras, fontsize=9)
    ax.set_ylabel(feat_label, fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", lw=0.4, alpha=0.4)


def _legend_handles():
    return [mpatches.Patch(color=COLOR_GT,  alpha=0.7, label="Ground truth"),
            mpatches.Patch(color=COLOR_GEN, alpha=0.7, label="Generated")]


def plot_overview(gt_rows, gen_rows, svaras, out_path, run_name, n_gen, summary=None):
    fig, axes = plt.subplots(4, 2, figsize=(12, 12.8), squeeze=False)
    fig.suptitle(f"GRU+VAE [{run_name}] — generated (n={n_gen}/svara) vs ground truth",
                 fontsize=12, fontweight="bold")
    for idx, (key, label) in enumerate(FEATURES):
        ax = axes[idx // 2][idx % 2]
        _violin_feature(ax, key, label, gt_rows, gen_rows, svaras, summary)
        if summary:
            d   = summary[key]["pooled"]["cohens_d"]
            tag = _effect_tag(d)
            ax.set_title(f"{label}  (pooled d={d:+.2f} [{tag}])", fontsize=8)
    fig.legend(handles=_legend_handles(), loc="lower center",
               ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare_gen] → {out_path}")


def plot_individual_features(gt_rows, gen_rows, svaras, out_dir, run_name, n_gen, summary=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, label in FEATURES:
        fig, ax = plt.subplots(figsize=(max(6, len(svaras) * 1.5), 4))
        _violin_feature(ax, key, label, gt_rows, gen_rows, svaras, summary)
        d_str = ""
        if summary:
            d = summary[key]["pooled"]["cohens_d"]
            d_str = f"  pooled d={d:+.2f} [{_effect_tag(d)}]"
        ax.set_title(f"[{run_name}] {label} — gen (n={n_gen}) vs GT{d_str}", fontsize=9)
        ax.legend(handles=_legend_handles(), fontsize=8)
        plt.tight_layout()
        p = out_dir / f"violin_{key.replace('_', '-')}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[compare_gen] → {p}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare GRU+VAE generated svaras vs ground truth."
    )
    parser.add_argument("--run",       default=None,
                        help="Run directory, e.g. run_001 (default: latest)")
    parser.add_argument("--n",         type=int, default=300,
                        help="Generated samples per svara label (default: 300)")
    parser.add_argument("--hard-mask", action="store_true",
                        help="Use musical grammar mask during generation")
    args = parser.parse_args()

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model    = load_model(args.run, device)
    run_name = args.run if args.run else sorted(CKPT_DIR.glob("run_*"))[-1].name

    mask_tag = "_hardmask" if args.hard_mask else ""
    out_dir  = settings.FIGURES_DIR / "gruvae" / run_name / f"compare_generated{mask_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── ground truth ──────────────────────────────────────────────────────────
    gt_rows, gt_seqs = load_gt_data(
        [s for s in SCALE_ORDER_PRESENT]
    )
    svaras = [s for s in SCALE_ORDER_PRESENT
              if any(r.get("svara_label") == s for r in gt_rows)]
    print(f"[compare_gen] GT: {len(gt_rows)} svaras  labels={svaras}")

    # ── generate (single pass, extract features + type seqs) ─────────────────
    print(f"[compare_gen] generating {args.n} × {len(svaras)} svaras "
          f"({'hard-mask' if args.hard_mask else 'soft'})...")
    gen_rows: list[dict] = []
    gen_seqs: dict[str, list[list[str]]] = {sv: [] for sv in svaras}
    for sv in svaras:
        samples = generate(model, sv, args.n, device, use_hard_mask=args.hard_mask)
        for sv_data in samples:
            feat = _feats_from_sample(sv_data)
            gen_seqs[sv].append(feat.pop("_type_seq"))
            gen_rows.append(feat)
    print(f"[compare_gen] generated: {len(gen_rows)} total")

    # ── numeric feature analysis ──────────────────────────────────────────────
    summary = compute_summary(gt_rows, gen_rows, svaras)
    print_and_save_summary(summary, svaras, out_dir, run_name, args.n)

    # ── proposals ─────────────────────────────────────────────────────────────
    propose_adjustments(summary, model.cfg, svaras, out_dir)

    # ── transition / succession analysis ─────────────────────────────────────
    plot_transition_analysis(gt_seqs, gen_seqs, svaras, out_dir, run_name, args.n)

    # ── violin plots ──────────────────────────────────────────────────────────
    plot_overview(gt_rows, gen_rows, svaras,
                  out_path=out_dir / "overview.png",
                  run_name=run_name, n_gen=args.n, summary=summary)
    plot_individual_features(gt_rows, gen_rows, svaras,
                             out_dir=out_dir / "violins",
                             run_name=run_name, n_gen=args.n, summary=summary)


if __name__ == "__main__":
    main()
