"""
Full synthesis pipeline: GRU+VAE structural sequence → per-segment pitch curves.

Segment curve models
--------------------
  CP  : CPVAE  — deviation around mean_cents (64-sample canonical curve)
  STA : ParamGRUVAE (or CurveModel fallback) — 0→1 tanh+osc, scaled to endpoints
  TR  : ParamGRUVAE (or CurveModel fallback) — 0→1 tanh+osc, scaled to endpoints
  SIL : NaN

Endpoint resolution rules
--------------------------
  STA.end     = STA.cents  (peak value from GRU+VAE)
  STA.start   = prev segment end
  TR.start    = prev segment end
  TR→CP  end  = first sample of next CP curve (from CPVAE phase-1)
  TR→STA end  = STA.cents
  TR→SIL end  = TR.start + delta  (sampled from GT TR→SIL distribution)
  TR (last)   : end sampled from GT ending_pitch_stats
  CP          : CPVAE handles deviations; ref pitch = mean_cents

Synthesis phases
----------------
  1 — Generate all CP curves (CPVAE) so their endpoints are known.
  2 — Resolve start/end cents for every segment (endpoint pre-pass).
  3 — Generate (k, s, A) for STA/TR via ParamGRUVAE (autoregressive, B2-aware)
        or sample independently from CurveModel if no checkpoint.
  4 — Render actual pitch curves from params.

Usage
-----
    python -m src.models.synthesis.synthesize --svara S --n 2
    python -m src.models.synthesis.synthesize --all --n 1 --dur 1.2
"""

from __future__ import annotations

import argparse
import dataclasses
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
from src.models.curve_vae.fit_sta_tr_curves import curve_model as _param_curve_fn
from src.models.param_gru.model_param_gru import ParamGRUConfig, ParamGRUVAE
from src.models.param_gru.dataset_param_gru import SVARA_TO_IDX as _PARAM_SVARA_IDX
from src.models.synthesis.ending_pitch_stats import (
    load_stats, load_tr_sil_stats,
    sample_ending_cents, sample_tr_sil_delta,
)

GRUVAE_DIR    = S.DATA_INTERIM / "models" / "gruvae_v4"
CPVAE_DIR     = S.DATA_INTERIM / "models" / "curve_vae" / "cp_vae_runs"
PARAM_GRU_DIR = S.DATA_INTERIM / "models" / "param_gru"
STA_TR_FIT    = S.DATA_INTERIM / "models" / "curve_vae" / "gt_curves_fitted.parquet"
OUT_DIR       = S.FIGURES_DIR / "synthesis"

TYPE_NAMES    = ["CP", "SIL", "STAp", "STAt", "TRa", "TRd"]
_TYPE_TO_IDX  = {"CP": 0, "SIL": 1, "STAp": 2, "STAt": 3, "TRa": 4, "TRd": 5}
SYNTH_SR      = 200   # output samples per second


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


def load_param_gruvae(device: torch.device) -> ParamGRUVAE | None:
    """Load latest ParamGRUVAE checkpoint, or None if not yet trained."""
    runs = sorted(PARAM_GRU_DIR.glob("run_*"))
    if not runs:
        return None
    ckpt_path = runs[-1] / "best.pt"
    if not ckpt_path.exists():
        return None
    ckpt  = torch.load(ckpt_path, map_location=device)
    known = {f.name for f in dataclasses.fields(ParamGRUConfig)}
    cfg   = ParamGRUConfig(**{k: v for k, v in ckpt["cfg"].items() if k in known})
    model = ParamGRUVAE(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[param_gru] {runs[-1].name}  ep={ckpt['epoch']}")
    return model


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
            cents = 0.0 if typ in ("SIL", "TRa", "TRd") else float(s[gruvae.cfg.type_dim + 1]) * 1200.0
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
    param_gruvae:  ParamGRUVAE | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """
    Generate one svara pitch contour (four phases).

    Phase 1 — Generate all CP curves (CPVAE) so their endpoints are known.
    Phase 2 — Resolve start/end cents for every segment (endpoint pre-pass).
    Phase 3 — Generate (k, s, A) for STA/TR via ParamGRUVAE (if available)
               or sample from CurveModel Gaussian (fallback).
    Phase 4 — Render pitch arrays from params.

    Returns
    -------
    pitch_cents : np.ndarray  — pitch in cents (NaN = silence)
    segments    : list[dict]  — enriched with dur_sec, start_cents, end_cents
    """
    if rng is None:
        rng = np.random.default_rng()

    # ── structural sequence ──────────────────────────────────────────────────
    structures = _generate_structure(gruvae, svara_label, 1, device, total_dur_sec, use_hard_mask)
    segs       = [dict(s) for s in structures[0]["segments"]]
    total_rel  = sum(s["dur_rel"] for s in segs) or 1.0
    for s in segs:
        s["dur_sec"] = s["dur_rel"] / total_rel * total_dur_sec

    # ── phase 1: generate all CP curves ─────────────────────────────────────
    sv_t = torch.tensor([SVARA_TO_IDX[svara_label]], dtype=torch.long, device=device)
    cp_curves: dict[int, np.ndarray] = {}

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

    # ── phase 2: resolve endpoints for all segments ──────────────────────────
    prev_end = 0.0
    for i, seg in enumerate(segs):
        next_seg = segs[i + 1] if i + 1 < len(segs) else None
        is_last  = next_seg is None

        if seg["type"] == "SIL":
            seg["start_cents"] = 0.0
            seg["end_cents"]   = 0.0
            prev_end = 0.0

        elif seg["type"] == "CP":
            curve = cp_curves[i]
            seg["start_cents"] = float(curve[0])
            seg["end_cents"]   = float(curve[-1])
            prev_end = seg["end_cents"]

        elif seg["type"] in ("STAp", "STAt"):
            seg["start_cents"] = prev_end
            seg["end_cents"]   = seg["cents"]
            prev_end = seg["end_cents"]

        elif seg["type"] in ("TRa", "TRd"):
            seg["start_cents"] = prev_end
            if is_last:
                end = sample_ending_cents(ending_stats, svara_label, seg["type"], rng)
            elif next_seg["type"] == "CP":
                end = float(cp_curves[i + 1][0])
            elif next_seg["type"] == "SIL":
                end = prev_end + sample_tr_sil_delta(tr_sil_stats, svara_label, rng)
            elif next_seg["type"] in ("STAp", "STAt"):
                end = next_seg["cents"]
            else:
                end = prev_end
            seg["end_cents"] = end
            prev_end = end

    # ── phase 3: generate (k, s, A) for STA/TR ──────────────────────────────
    seg_params: dict[int, tuple[float, float, float]] = {}

    if param_gruvae is not None:
        # ParamGRUVAE: autoregressive, B2-slope-aware
        n_segs   = len(segs)
        total_dur = sum(s["dur_sec"] for s in segs) or 1.0

        seg_oh    = np.zeros((1, n_segs, 6), dtype=np.float32)
        dur_rel_t = np.zeros((1, n_segs), dtype=np.float32)
        delta_n   = np.zeros((1, n_segs), dtype=np.float32)
        dur_sec_t = np.zeros((1, n_segs), dtype=np.float32)
        delta_c   = np.zeros((1, n_segs), dtype=np.float32)
        sta_tr_m  = np.zeros((1, n_segs), dtype=bool)

        for i, seg in enumerate(segs):
            seg_oh[0, i, _TYPE_TO_IDX[seg["type"]]] = 1.0
            dur_rel_t[0, i] = seg["dur_sec"] / total_dur
            d = seg["end_cents"] - seg["start_cents"]
            delta_n[0, i]   = d / 1200.0
            dur_sec_t[0, i] = seg["dur_sec"]
            delta_c[0, i]   = d
            sta_tr_m[0, i]  = seg["type"] in ("STAp", "STAt", "TRa", "TRd")

        with torch.no_grad():
            result = param_gruvae.generate(
                seg_type_oh = torch.from_numpy(seg_oh).to(device),
                dur_rel     = torch.from_numpy(dur_rel_t).to(device),
                delta_norm  = torch.from_numpy(delta_n).to(device),
                dur_sec     = torch.from_numpy(dur_sec_t).to(device),
                delta_cents = torch.from_numpy(delta_c).to(device),
                sta_tr_mask = torch.from_numpy(sta_tr_m).to(device),
                lengths     = torch.tensor([n_segs], device=device),
                svara_idx   = torch.tensor([_PARAM_SVARA_IDX[svara_label]], device=device),
                total_dur   = torch.tensor([total_dur], device=device),
            )
        params = result["params"][0]  # (n_segs, 3)
        k_all, s_all, A_all = param_gruvae.decode_params(params)
        for i, seg in enumerate(segs):
            if seg["type"] in ("STAp", "STAt", "TRa", "TRd"):
                seg_params[i] = (float(k_all[i]), float(s_all[i]), float(A_all[i]))

    else:
        # Fallback: sample independently from CurveModel Gaussian
        for i, seg in enumerate(segs):
            if seg["type"] in ("STAp", "STAt", "TRa", "TRd"):
                gen = curve_model.generate(seg["type"], svara=svara_label, n=1, rng=rng)[0]
                # Extract params from generated curve_fn by fitting isn't easy;
                # use the CurveModel's sampled params directly
                seg_params[i] = (gen["k"], gen["s"], gen["A"])

    # ── phase 4: render pitch curves ─────────────────────────────────────────
    all_curves: list[np.ndarray] = []

    for i, seg in enumerate(segs):
        n = max(1, round(seg["dur_sec"] * SYNTH_SR))

        if seg["type"] == "SIL":
            all_curves.append(np.full(n, np.nan))

        elif seg["type"] == "CP":
            all_curves.append(cp_curves[i])

        elif seg["type"] in ("STAp", "STAt", "TRa", "TRd"):
            start = seg["start_cents"]
            end   = seg["end_cents"]
            delta = end - start
            if i in seg_params and abs(delta) > 1e-6:
                k, s, A = seg_params[i]
                t    = np.linspace(0.0, 1.0, n)
                norm = _param_curve_fn(t, k, s, A)
            else:
                norm = np.linspace(0.0, 1.0, n)
            all_curves.append(start + norm * delta)

    pitch_cents = np.concatenate(all_curves) if all_curves else np.array([])
    return pitch_cents, segs


# ── plotting ──────────────────────────────────────────────────────────────────

SEG_COLOR = {"CP": "#4caf50", "SIL": "#9e9e9e", "STAp": "#ff9800", "STAt": "#e65100",
             "TRa": "#2196f3", "TRd": "#0277bd"}


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
    param_gruvae  = load_param_gruvae(device)
    ending_stats  = load_stats()
    tr_sil_stats  = load_tr_sil_stats()
    rng           = np.random.default_rng(42)

    if param_gruvae is None:
        print("[param_gru] no checkpoint found — falling back to CurveModel Gaussian sampling")

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
                param_gruvae=param_gruvae,
            )
            seg_str = " ".join(s["type"] for s in segs)
            _plot_one(axes[r][c], pitch, segs, f"{svara} #{c+1} [{seg_str}]")

    # legend (segment type colours) — deduplicate
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=SEG_COLOR[t], alpha=0.5, label=t)
               for t in ["CP", "STAp", "STAt", "TRa", "TRd", "SIL"]]
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
    plt.show()


if __name__ == "__main__":
    main()
