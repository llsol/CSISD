"""
Full synthesis pipeline: GRU+VAE structural sequence → per-segment pitch curves.

Segment curve models
--------------------
  CP  : CPVAE  — deviation around mean_cents (64-sample canonical curve)
  STA : ParamGRUVAE (or CurveModel fallback) — 0→1 tanh+osc, scaled to endpoints
  TR  : ParamGRUVAE (or CurveModel fallback) — 0→1 tanh+osc, scaled to endpoints
  SIL : NaN

Endpoint resolution rules (C0)
--------------------------
  STA.end     = STA.cents  (peak value from GRU+VAE)
  STA→CP end  = first sample of next CP curve (override STA.cents)
  STA.start   = prev segment end
  TR.start    = prev segment end
  TR→CP  end  = first sample of next CP curve (from CPVAE phase-1)
  TR→STA end  = STA.cents
  TR→SIL end  = TR.start + delta  (sampled from GT TR→SIL distribution)
  TR (last)   : end sampled from GT ending_pitch_stats
  CP          : CPVAE handles deviations; ref pitch = mean_cents

Derivative continuity rules (C1)
---------------------------------
  CP→STA/TR   : dy0_required = CP end-slope  (physical finite-diff → normalized)
  STA/TR→CP   : m1_required  = CP start-slope (→ analytic A adjustment)
  STA/TR start (after SIL/svara start): dy0_required sampled from GT (BoundarySlopeSampler)
  TR→SIL / TR at end: m1_required sampled from GT (BoundarySlopeSampler)
  STA/TR→STA/TR: dy0 propagated auto-regressively through ParamGRU generate()

Synthesis phases
----------------
  1 — Generate all CP curves (CPVAE); compute their boundary slopes (¢/s).
  2 — Resolve start/end cents for every segment (endpoint pre-pass, C0).
  3 — Boundary slope pre-pass: fill dy0_init_arr and m1_req_dict (C1).
  4 — Generate (k, s, A) for STA/TR via ParamGRUVAE (autoregressive)
        or sample independently from CurveModel if no checkpoint.
        Analytic A adjustment applied where m1_req_dict is set.
  5 — Render actual pitch curves from params.

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
from src.models.curve_vae.fit_sta_tr_curves import curve_model as _param_curve_fn, A_MIN, A_MAX
from src.models.synthesis.boundary_slopes import (
    BoundarySlopeSampler, load_boundary_sampler, solve_A_for_m1,
)
from src.models.param_gru.model_param_gru import ParamGRUConfig, ParamGRU, ResidualDist, load_residuals
from src.models.param_gru.dataset_param_gru import SVARA_TO_IDX as _PARAM_SVARA_IDX
from src.models.synthesis.ending_pitch_stats import (
    load_stats, load_tr_sil_stats,
    sample_ending_cents, sample_tr_sil_delta,
)

GRUVAE_DIR    = S.GRUVAE_DIR
CPVAE_DIR     = S.CURVE_VAE_DIR / "cp_vae_runs"
PARAM_GRU_DIR = S.PARAM_GRU_DIR
STA_TR_FIT    = S.CURVE_VAE_DIR / "gt_curves_fitted.parquet"
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


def load_param_gruvae(device: torch.device) -> tuple[ParamGRU | None, dict[str, ResidualDist] | None]:
    """Load latest ParamGRU checkpoint and residual dists, or (None, None)."""
    runs = sorted(PARAM_GRU_DIR.glob("run_*"))
    if not runs:
        return None, None
    ckpt_path = runs[-1] / "best.pt"
    if not ckpt_path.exists():
        return None, None
    ckpt  = torch.load(ckpt_path, map_location=device)
    known = {f.name for f in dataclasses.fields(ParamGRUConfig)}
    cfg   = ParamGRUConfig(**{k: v for k, v in ckpt["cfg"].items() if k in known})
    model = ParamGRU(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[param_gru] {runs[-1].name}  ep={ckpt['epoch']}")

    res_path = runs[-1] / "residuals.npz"
    dists = load_residuals(res_path) if res_path.exists() else None
    if dists:
        print(f"[param_gru] residual dists: {sorted(dists.keys())}")
    else:
        print("[param_gru] no residuals.npz — generation will be deterministic")
    return model, dists


# ── structural generation ─────────────────────────────────────────────────────

def _generate_structure(
    gruvae:        SvaraGRUVAE,
    svara_label:   str,
    n:             int,
    device:        torch.device,
    use_hard_mask: bool,
) -> list[dict]:
    """Returns list of n svara dicts, each with 'segments' list."""
    idx_t = torch.tensor([SVARA_TO_IDX[svara_label]] * n, dtype=torch.long, device=device)
    with torch.no_grad():
        out = gruvae.generate(
            batch_size=n, svara_idx=idx_t, device=device,
            use_hard_mask=use_hard_mask,
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
            segments.append({"type": typ, "dur_abs_sec": float(s[gruvae.cfg.type_dim]), "cents": cents})
        results.append({"svara": svara_label, "segments": segments})
    return results


# ── helpers ───────────────────────────────────────────────────────────────────

def _resample(arr: np.ndarray, n: int) -> np.ndarray:
    if len(arr) == n:
        return arr
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(arr)), arr)


# ── pitch anchor helper ───────────────────────────────────────────────────────

def _sample_start_cents(
    df_curves: pl.DataFrame,
    svara: str,
    seg_type: str,
    rng: np.random.Generator,
) -> float:
    """Sample starting pitch from GT p_start_cents for (svara, seg_type)."""
    sub = df_curves.filter(
        (pl.col("svara_label") == svara) & (pl.col("seg_type") == seg_type)
    )["p_start_cents"].drop_nulls().to_numpy()
    if len(sub) == 0:
        sub = df_curves.filter(
            pl.col("seg_type") == seg_type
        )["p_start_cents"].drop_nulls().to_numpy()
    if len(sub) == 0:
        return 0.0
    return float(rng.choice(sub))


# ── full svara synthesis ──────────────────────────────────────────────────────

def synthesize_svara(
    svara_label:   str,
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
    param_gruvae:      ParamGRU | None = None,
    residual_dists:    dict | None = None,
    df_curves:         pl.DataFrame | None = None,
    boundary_sampler:  BoundarySlopeSampler | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """
    Generate one svara pitch contour (five phases).

    Phase 1 — Generate all CP curves (CPVAE); compute boundary slopes.
    Phase 2 — Resolve start/end cents for every segment (C0 continuity).
    Phase 3 — Boundary slope pre-pass: fill dy0_init_arr / m1_req_dict (C1).
    Phase 4 — Generate (k, s, A) for STA/TR via ParamGRUVAE (if available)
               or sample from CurveModel Gaussian (fallback).
    Phase 5 — Render pitch arrays from params.

    Returns
    -------
    pitch_cents : np.ndarray  — pitch in cents (NaN = silence)
    segments    : list[dict]  — enriched with dur_sec, start_cents, end_cents
    """
    if rng is None:
        rng = np.random.default_rng()

    # ── structural sequence ──────────────────────────────────────────────────
    structures = _generate_structure(gruvae, svara_label, 1, device, use_hard_mask)
    segs       = [dict(s) for s in structures[0]["segments"]]
    for s in segs:
        s["dur_sec"] = s["dur_abs_sec"]

    # ── phase 1: generate all CP curves; compute boundary slopes (¢/s) ─────
    sv_t = torch.tensor([SVARA_TO_IDX[svara_label]], dtype=torch.long, device=device)
    cp_curves:  dict[int, np.ndarray] = {}
    cp_m0_phys: dict[int, float]      = {}   # physical slope at CP start (¢/s)
    cp_m1_phys: dict[int, float]      = {}   # physical slope at CP end   (¢/s)

    _SLOPE_WIN = 4   # samples for finite-diff slope estimate

    for i, seg in enumerate(segs):
        if seg["type"] != "CP":
            continue
        n     = max(1, round(seg["dur_sec"] * SYNTH_SR))
        dur_t = torch.tensor([seg["dur_sec"]], dtype=torch.float32, device=device)
        dev64 = (
            cpvae.generate(1, sv_t, dur_t, scale=scale, verbose=verbose)
            .cpu().numpy()[0] * scale
        )
        curve = seg["cents"] + _resample(dev64, n)
        cp_curves[i] = curve

        win = min(_SLOPE_WIN, len(curve) - 1)
        if win > 0:
            cp_m0_phys[i] = (curve[win]  - curve[0])       / (win / SYNTH_SR)
            cp_m1_phys[i] = (curve[-1]   - curve[-(win+1)]) / (win / SYNTH_SR)
        else:
            cp_m0_phys[i] = 0.0
            cp_m1_phys[i] = 0.0

    # ── phase 2: resolve endpoints for all segments ──────────────────────────
    # Initialise prev_end from GT start-pitch distribution for the first
    # non-CP, non-SIL segment (avoids the unrealistic 0 ¢ anchor).
    _first_voiced = next(
        (s for s in segs if s["type"] not in ("SIL", "CP")), None
    )
    prev_end = (
        _sample_start_cents(df_curves, svara_label, _first_voiced["type"], rng)
        if (df_curves is not None and _first_voiced is not None)
        else 0.0
    )

    for i, seg in enumerate(segs):
        next_seg = segs[i + 1] if i + 1 < len(segs) else None
        is_last  = next_seg is None

        if seg["type"] == "SIL":
            seg["start_cents"] = 0.0
            seg["end_cents"]   = 0.0
            # After silence the pitch restarts — sample from GT distribution
            # for whichever voiced segment comes next.
            _next_voiced = next(
                (segs[j] for j in range(i + 1, len(segs))
                 if segs[j]["type"] != "SIL"), None
            )
            if (df_curves is not None
                    and _next_voiced is not None
                    and _next_voiced["type"] != "CP"):
                prev_end = _sample_start_cents(
                    df_curves, svara_label, _next_voiced["type"], rng
                )
            else:
                prev_end = 0.0

        elif seg["type"] == "CP":
            curve = cp_curves[i]
            seg["start_cents"] = float(curve[0])
            seg["end_cents"]   = float(curve[-1])
            prev_end = seg["end_cents"]

        elif seg["type"] in ("STAp", "STAt"):
            seg["start_cents"] = prev_end
            # C0: if next seg is CP, force end to CP's actual first sample
            if next_seg is not None and next_seg["type"] == "CP":
                seg["end_cents"] = float(cp_curves[i + 1][0])
            else:
                seg["end_cents"] = seg["cents"]
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

    # ── boundary slope pre-pass (C1) ────────────────────────────────────────
    # For each STA/TR we determine two slope constraints:
    #   dy0_init_arr[i]  — required normalized start slope (raw, tanh applied in generate())
    #   m1_req_dict[i]   — required normalized end slope (→ analytic A adjustment)
    #
    # Priority for dy0 (start):
    #   CP → STA/TR  : CP end slope      (physical finite-diff, converted to normalized)
    #   SIL / start  : GT distribution   (BoundarySlopeSampler, Case B)
    #   STA/TR→STA/TR: auto-regressive   (propagated inside generate(), no init)
    #
    # Priority for m1 (end):
    #   STA/TR → CP  : CP start slope    (physical finite-diff, converted to normalized)
    #   TR → SIL/end : GT distribution   (BoundarySlopeSampler, Case A)
    n_segs = len(segs)
    dy0_init_arr  = np.zeros(n_segs, dtype=np.float32)
    m1_req_dict: dict[int, float] = {}

    _STA_TR_TYPES = {"STAp", "STAt", "TRa", "TRd"}

    for i, seg in enumerate(segs):
        stype    = seg["type"]
        if stype not in _STA_TR_TYPES:
            continue
        delta_i  = seg["end_cents"] - seg["start_cents"]
        dur_i    = seg["dur_sec"]
        prev_seg = segs[i - 1] if i > 0 else None
        next_seg = segs[i + 1] if i + 1 < n_segs else None

        # ── dy0: start-slope requirement ──────────────────────────────────
        if prev_seg is not None and prev_seg["type"] == "CP":
            m1_cp = cp_m1_phys.get(i - 1, 0.0)
            if abs(delta_i) > 1.0 and dur_i > 1e-4:
                dy0_init_arr[i] = m1_cp * dur_i / delta_i
        elif boundary_sampler is not None and (
            prev_seg is None or prev_seg["type"] == "SIL"
        ):
            dy0_init_arr[i] = boundary_sampler.sample_dy0_for_boundary_start(
                stype, svara_label, delta_i, dur_i, rng,
            )
        # else: preceded by STA/TR → auto-regressive chain in generate()

        # ── m1: end-slope requirement ──────────────────────────────────────
        if next_seg is not None and next_seg["type"] == "CP":
            m0_cp = cp_m0_phys.get(i + 1, 0.0)
            if abs(delta_i) > 1.0 and dur_i > 1e-4:
                m1_req_dict[i] = m0_cp * dur_i / delta_i
        elif boundary_sampler is not None and stype in ("TRa", "TRd"):
            is_end_boundary = next_seg is None or next_seg["type"] == "SIL"
            if is_end_boundary:
                next_voiced = next(
                    (segs[j] for j in range(i + 1, n_segs)
                     if segs[j]["type"] != "SIL"),
                    None,
                )
                if next_voiced is not None and next_voiced["type"] in _STA_TR_TYPES:
                    delta_succ = next_voiced["end_cents"] - next_voiced["start_cents"]
                    dur_succ   = next_voiced["dur_sec"]
                    m1_req = boundary_sampler.compute_m1_from_successor(
                        next_voiced["type"], svara_label,
                        delta_succ, dur_succ,
                        delta_i, dur_i, rng,
                    )
                else:
                    m1_req = boundary_sampler.sample_m1_for_boundary_end(
                        stype, svara_label, rng,
                    )
                if m1_req is not None:
                    m1_req_dict[i] = m1_req

    # ── phase 4: generate (k, s, A) for STA/TR ──────────────────────────────
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

        dy0_init_t = torch.from_numpy(dy0_init_arr[np.newaxis, :]).to(device)  # (1, n_segs)

        with torch.no_grad():
            result = param_gruvae.generate(
                seg_type_oh       = torch.from_numpy(seg_oh).to(device),
                dur_rel           = torch.from_numpy(dur_rel_t).to(device),
                delta_norm        = torch.from_numpy(delta_n).to(device),
                dur_sec           = torch.from_numpy(dur_sec_t).to(device),
                delta_cents       = torch.from_numpy(delta_c).to(device),
                sta_tr_mask       = torch.from_numpy(sta_tr_m).to(device),
                lengths           = torch.tensor([n_segs], device=device),
                svara_idx         = torch.tensor([_PARAM_SVARA_IDX[svara_label]], device=device),
                total_dur         = torch.tensor([total_dur], device=device),
                residual_dists    = residual_dists,
                rng               = rng,
                dy0_required_init = dy0_init_t,
            )
        params = result["params"][0]  # (n_segs, 3)
        k_all, s_all, A_all = param_gruvae.decode_params(params)
        for i, seg in enumerate(segs):
            if seg["type"] in ("STAp", "STAt", "TRa", "TRd"):
                k_i = float(k_all[i])
                s_i = float(s_all[i])
                A_i = float(A_all[i])
                # Cas A: ajust analític de A per a TR→SIL o TR al final
                if i in m1_req_dict:
                    A_i = solve_A_for_m1(k_i, s_i, m1_req_dict[i])
                seg_params[i] = (k_i, s_i, A_i)

    else:
        # Fallback: sample independently from CurveModel Gaussian
        for i, seg in enumerate(segs):
            if seg["type"] in ("STAp", "STAt", "TRa", "TRd"):
                gen = curve_model.generate(seg["type"], svara=svara_label, n=1, rng=rng)[0]
                k_i, s_i, A_i = gen["k"], gen["s"], gen["A"]
                if i in m1_req_dict:
                    A_i = solve_A_for_m1(k_i, s_i, m1_req_dict[i])
                seg_params[i] = (k_i, s_i, A_i)

    # ── phase 5: render pitch curves ─────────────────────────────────────────
    # Each segment represents a half-open interval [t_start, t_start + dur).
    # We drop the last sample of every segment except the final one to avoid
    # duplicating the boundary value (linspace includes both endpoints, so
    # curve[-1] == curve_next[0] without this drop).
    all_curves: list[np.ndarray] = []
    n_segs_total = len(segs)

    for i, seg in enumerate(segs):
        n       = max(1, round(seg["dur_sec"] * SYNTH_SR))
        is_last = (i == n_segs_total - 1)

        if seg["type"] == "SIL":
            curve = np.full(n, np.nan)

        elif seg["type"] == "CP":
            curve = cp_curves[i]

        elif seg["type"] in ("STAp", "STAt", "TRa", "TRd"):
            start = seg["start_cents"]
            delta = seg["end_cents"] - start
            if i in seg_params and abs(delta) > 1e-6:
                k, s, A = seg_params[i]
                t    = np.linspace(0.0, 1.0, n)
                norm = _param_curve_fn(t, k, s, A)
            else:
                norm = np.linspace(0.0, 1.0, n)
            curve = start + norm * delta

        else:
            curve = np.full(n, np.nan)

        all_curves.append(curve if is_last else curve[:-1])

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
    total_dur = sum(s["dur_sec"] for s in segments)
    ax.set_title(f"{title}  [{total_dur:.2f}s]", fontsize=8)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize full pitch contours from GRU+VAE.")
    parser.add_argument("--svara",     default="S",  choices=SVARA_LABELS)
    parser.add_argument("--all",       action="store_true")
    parser.add_argument("--n",         type=int,   default=2)
    parser.add_argument("--run",       default=None, help="GRU+VAE run dir")
    parser.add_argument("--no-hard-mask", dest="hard_mask", action="store_false",
                        help="Disable grammar mask (default: mask ON)")
    parser.set_defaults(hard_mask=True)
    parser.add_argument("--verbose",   action="store_true",
                        help="Print CP VAE accept rate per segment")
    parser.add_argument("--out",       default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gruvae                  = load_gruvae(args.run, device)
    cpvae, scale            = load_cpvae(device)
    curve_model             = load_curve_model()
    param_gruvae, res_dists = load_param_gruvae(device)
    ending_stats            = load_stats()
    tr_sil_stats            = load_tr_sil_stats()
    df_curves               = pl.read_parquet(STA_TR_FIT) if STA_TR_FIT.exists() else None
    rng                     = np.random.default_rng(42)
    try:
        boundary_sampler = load_boundary_sampler(curve_model)
        print("[boundary] sampler loaded")
    except Exception as e:
        boundary_sampler = None
        print(f"[boundary] sampler unavailable: {e}")

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
                residual_dists=res_dists,
                df_curves=df_curves,
                boundary_sampler=boundary_sampler,
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
        f"Synthesis — svara {'all' if args.all else args.svara}",
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
