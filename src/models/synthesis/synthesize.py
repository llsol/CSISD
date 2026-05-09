"""
Full synthesis pipeline: GRU+VAE structural sequence → per-segment pitch curves.

Segment curve models
--------------------
  CP  : CPVAE  — deviation around mean_cents (64-sample canonical curve)
  STA : CurveModel("STA") — normalized 0→1 curve scaled to [start, end] cents
  TR  : CurveModel("TR")  — normalized 1→0 curve scaled to [start, end] cents
  SIL : zeros

Endpoint resolution rules (per user spec)
-----------------------------------------
  STA.end     = STA.cents  (peak value from GRU+VAE)
  STA.start   = prev segment end
  TR.start    = prev segment end  (= STA peak or CP mean)
  TR→CP  end  = CP.mean_cents  (rule 1.1.1)
  TR→STA end  = STA.cents
  TR→SIL end  = TR.start + delta   (delta sampled from GT TR→SIL distribution)
  TR last seg : end sampled from GT ending_pitch_stats
  TR.start    = prev segment end
  Last TR/STA : ending pitch sampled from GT distribution (ending_pitch_stats.py)
  CP          : CPVAE handles deviations; ref pitch = mean_cents

Usage
-----
    python -m src.models.synthesis.synthesize --svara S --n 2
    python -m src.models.synthesis.synthesize --all --n 1 --dur 1.2
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields as dataclass_fields
from pathlib import Path

import numpy as np
import polars as pl
import torch
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import settings as S
from src.models.gruvae.dataset_gruvae import SVARA_LABELS, SVARA_TO_IDX
from src.models.gruvae.model_gruvae import ModelConfig, SvaraGRUVAE
from src.models.curve_vae.cp_vae import CPVAE, CPVAEConfig
from src.models.curve_vae.sta_tr_model import CurveModel
from src.models.synthesis.ending_pitch_stats import (
    load_stats, load_tr_sil_stats,
    sample_ending_cents, sample_tr_sil_delta,
)

GRUVAE_DIR = S.DATA_INTERIM / "models" / "gruvae_v4"
CPVAE_DIR  = S.DATA_INTERIM / "models" / "curve_vae" / "cp_vae_runs"
STA_TR_FIT = S.DATA_INTERIM / "models" / "curve_vae" / "gt_curves_fitted.parquet"
OUT_DIR    = S.FIGURES_DIR / "synthesis"

TYPE_NAMES = ["CP", "SIL", "STA", "TR"]
SYNTH_SR   = 200   # output samples per second


# ── model loading ─────────────────────────────────────────────────────────────

def load_gruvae(run: str | None, device: torch.device) -> SvaraGRUVAE:
    run_dir = (GRUVAE_DIR / run) if run else sorted(GRUVAE_DIR.glob("run_*"))[-1]
    ckpt    = torch.load(run_dir / "best.pt", map_location=device)
    known   = {f.name for f in dataclass_fields(ModelConfig)}
    cfg     = ModelConfig(**{k: v for k, v in ckpt["model_cfg"].items() if k in known})
    cfg.__dict__.setdefault("use_attention", False)
    model   = SvaraGRUVAE(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[gruvae]   {run_dir.name}  ep={ckpt['epoch']}")
    return model


def load_cpvae(device: torch.device) -> tuple[CPVAE, float]:
    run_dir = sorted(CPVAE_DIR.glob("run_*"))[-1]
    best    = run_dir / "best.pt"
    ckpt    = torch.load(best, map_location=device)
    model   = CPVAE(CPVAEConfig(**ckpt["cfg"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    scale = float(ckpt["scale"])
    print(f"[cp_vae]   {run_dir.name}  ep={ckpt['epoch']}  scale={scale:.2f}¢")
    return model, scale


def load_curve_model() -> CurveModel:
    df_fit = pl.read_parquet(STA_TR_FIT)
    return CurveModel().fit(df_fit)


# ── structural generation ─────────────────────────────────────────────────────

def _generate_structure(
    gruvae:        SvaraGRUVAE,
    svara_label:   str,
    n:             int,
    device:        torch.device,
    total_dur_sec: float | None,
    use_hard_mask: bool,
) -> list[dict]:
    """Returns list of n svara dicts, each with 'segments' list."""
    idx_t = torch.tensor([SVARA_TO_IDX[svara_label]] * n, dtype=torch.long, device=device)
    dur_t = (
        torch.full((n,), float(total_dur_sec), dtype=torch.float32, device=device)
        if total_dur_sec is not None else None
    )
    with torch.no_grad():
        out = gruvae.generate(
            batch_size=n, svara_idx=idx_t, device=device,
            total_dur=dur_t, use_hard_mask=use_hard_mask,
        )
    gen       = out["generated"]
    pred_lens = out["pred_length"].float().cpu()

    results = []
    for i in range(n):
        n_segs = int(round(float(pred_lens[i])))
        n_segs = max(1, min(n_segs, gruvae.cfg.max_seq_len))
        segs   = gen[i, :n_segs].cpu().numpy()

        segments = []
        for s in segs:
            typ   = TYPE_NAMES[int(np.argmax(s[:gruvae.cfg.type_dim]))]
            cents = 0.0 if typ in ("SIL", "TR") else float(s[gruvae.cfg.type_dim + 1]) * 1200.0
            segments.append({"type": typ, "dur_rel": float(s[gruvae.cfg.type_dim]), "cents": cents})
        results.append({"svara": svara_label, "segments": segments})
    return results


# ── helpers ───────────────────────────────────────────────────────────────────

def _resample(arr: np.ndarray, n: int) -> np.ndarray:
    if len(arr) == n:
        return arr
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(arr)), arr)


# ── full svara synthesis ──────────────────────────────────────────────────────

def synthesize_svara(
    svara_label:   str,
    total_dur_sec: float,
    gruvae:        SvaraGRUVAE,
    cpvae:         CPVAE,
    scale:         float,
    curve_model:   CurveModel,
    ending_stats:  pl.DataFrame,
    tr_sil_stats:  pl.DataFrame,
    device:        torch.device,
    use_hard_mask: bool = True,
    rng:           np.random.Generator | None = None,
    verbose:       bool = False,
) -> tuple[np.ndarray, list[dict]]:
    """
    Generate one svara pitch contour (two-phase).

    Phase 1 — generate all CP curves first (CPVAE gives actual first/last sample).
    Phase 2 — generate STA / TR / SIL in chronological order, using actual CP
              boundary values so continuity is exact.

    Returns
    -------
    pitch_cents : np.ndarray, shape (n_samples,)  — pitch in cents (NaN = silence)
    segments    : list[dict] — segment dicts enriched with dur_sec, start_cents, end_cents
    """
    if rng is None:
        rng = np.random.default_rng()

    # ── structural sequence ──────────────────────────────────────────────────
    structures = _generate_structure(gruvae, svara_label, 1, device, total_dur_sec, use_hard_mask)
    segs       = [dict(s) for s in structures[0]["segments"]]

    total_rel = sum(s["dur_rel"] for s in segs) or 1.0
    for s in segs:
        s["dur_sec"] = s["dur_rel"] / total_rel * total_dur_sec

    # ── phase 1: generate all CP curves ─────────────────────────────────────
    sv_t = torch.tensor([SVARA_TO_IDX[svara_label]], dtype=torch.long,  device=device)
    cp_curves: dict[int, np.ndarray] = {}   # index → pitch array (cents)

    for i, seg in enumerate(segs):
        if seg["type"] != "CP":
            continue
        n     = max(1, round(seg["dur_sec"] * SYNTH_SR))
        dur_t = torch.tensor([seg["dur_sec"]], dtype=torch.float32, device=device)
        dev64 = (
            cpvae.generate(1, sv_t, dur_t, scale=scale, verbose=verbose)
            .cpu().numpy()[0] * scale
        )
        cp_curves[i] = seg["cents"] + _resample(dev64, n)

    # ── phase 2: generate all segments in linear order ───────────────────────
    all_curves: list[np.ndarray] = []
    prev_end = 0.0   # last actual pitch sample of previous segment

    for i, seg in enumerate(segs):
        n        = max(1, round(seg["dur_sec"] * SYNTH_SR))
        next_seg = segs[i + 1] if i + 1 < len(segs) else None
        is_last  = next_seg is None

        if seg["type"] == "SIL":
            seg["start_cents"] = 0.0
            seg["end_cents"]   = 0.0
            all_curves.append(np.full(n, np.nan))
            prev_end = 0.0

        elif seg["type"] == "CP":
            curve = cp_curves[i]
            seg["start_cents"] = float(curve[0])
            seg["end_cents"]   = float(curve[-1])
            all_curves.append(curve)
            prev_end = seg["end_cents"]

        elif seg["type"] == "STA":
            # STA.start = prev segment end; STA.end = STA.cents (peak, from GRU+VAE)
            start = prev_end
            end   = seg["cents"]
            t     = np.linspace(0.0, 1.0, n)
            norm  = curve_model.generate("STA", svara=svara_label, n=1, rng=rng)[0]["curve_fn"](t)
            curve = start + norm * (end - start)
            seg["start_cents"] = start
            seg["end_cents"]   = end
            all_curves.append(curve)
            prev_end = end

        elif seg["type"] == "TR":
            start = prev_end   # = STA.cents or CP.last (actual)

            # determine TR end from next segment
            if is_last:
                end = sample_ending_cents(ending_stats, svara_label, "TR", rng)
            elif next_seg["type"] == "CP":
                # TR.end = first actual sample of next CP (from phase-1 curve)
                next_cp_curve = cp_curves[i + 1]
                end = float(next_cp_curve[0])
            elif next_seg["type"] == "SIL":
                end = start + sample_tr_sil_delta(tr_sil_stats, svara_label, rng)
            elif next_seg["type"] == "STA":
                end = next_seg["cents"]   # STA peak
            else:
                end = prev_end   # TR→TR: grammar forbids, fallback

            t    = np.linspace(0.0, 1.0, n)
            norm = curve_model.generate("TR", svara=svara_label, n=1, rng=rng)[0]["curve_fn"](t)
            # TR norm goes 1→0; affine so curve[0]=start, curve[-1]=end
            curve = end + norm * (start - end)
            seg["start_cents"] = start
            seg["end_cents"]   = end
            all_curves.append(curve)
            prev_end = end

    pitch_cents = np.concatenate(all_curves) if all_curves else np.array([])
    return pitch_cents, segs


# ── plotting ──────────────────────────────────────────────────────────────────

SEG_COLOR = {"CP": "#4caf50", "SIL": "#9e9e9e", "STA": "#ff9800", "TR": "#2196f3"}


def _plot_one(ax: plt.Axes, pitch: np.ndarray, segments: list[dict], title: str) -> None:
    t = np.arange(len(pitch)) / SYNTH_SR

    # shade segments
    cursor = 0
    for seg in segments:
        n = max(1, round(seg["dur_sec"] * SYNTH_SR))
        ax.axvspan(cursor / SYNTH_SR, (cursor + n) / SYNTH_SR,
                   alpha=0.15, color=SEG_COLOR[seg["type"]], label=seg["type"])
        cursor += n

    ax.plot(t, pitch, color="#333", lw=0.8)
    ax.axhline(0,    color="steelblue", lw=0.5, ls="--", alpha=0.5)
    ax.set_xlabel("time (s)", fontsize=7)
    ax.set_ylabel("cents", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_title(title, fontsize=8)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize full pitch contours from GRU+VAE.")
    parser.add_argument("--svara",     default="S",  choices=SVARA_LABELS)
    parser.add_argument("--all",       action="store_true")
    parser.add_argument("--n",         type=int,   default=2)
    parser.add_argument("--dur",       type=float, default=1.0,
                        help="Total svara duration in seconds")
    parser.add_argument("--run",       default=None, help="GRU+VAE run dir")
    parser.add_argument("--no-hard-mask", dest="hard_mask", action="store_false",
                        help="Disable grammar mask (default: mask ON)")
    parser.set_defaults(hard_mask=True)
    parser.add_argument("--verbose",   action="store_true",
                        help="Print CP VAE accept rate per segment")
    parser.add_argument("--out",       default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gruvae        = load_gruvae(args.run, device)
    cpvae, scale  = load_cpvae(device)
    curve_model   = load_curve_model()
    ending_stats  = load_stats()
    tr_sil_stats  = load_tr_sil_stats()
    rng           = np.random.default_rng(42)

    svaras = SVARA_LABELS if args.all else [args.svara]
    n_rows = len(svaras)
    n_cols = args.n

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 2.5), squeeze=False)

    for r, svara in enumerate(svaras):
        for c in range(n_cols):
            pitch, segs = synthesize_svara(
                svara_label=svara,
                total_dur_sec=args.dur,
                gruvae=gruvae,
                cpvae=cpvae,
                scale=scale,
                curve_model=curve_model,
                ending_stats=ending_stats,
                tr_sil_stats=tr_sil_stats,
                device=device,
                use_hard_mask=args.hard_mask,
                rng=rng,
                verbose=args.verbose,
            )
            seg_str = " ".join(s["type"] for s in segs)
            _plot_one(axes[r][c], pitch, segs, f"{svara} #{c+1} [{seg_str}]")

    # legend (segment type colours) — deduplicate
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=SEG_COLOR[t], alpha=0.5, label=t)
               for t in ["CP", "STA", "TR", "SIL"]]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        f"Synthesis — svara {'all' if args.all else args.svara}  dur={args.dur}s",
        fontsize=10, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.04, 1, 1])

    svara_tag = "all" if args.all else args.svara
    out = Path(args.out) if args.out else OUT_DIR / f"synth_{svara_tag}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"→ {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
