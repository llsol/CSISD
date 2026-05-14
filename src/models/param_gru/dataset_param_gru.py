"""
Dataset for Parameter-GRU: sequences of (k, s, A) curve parameters within svara occurrences.

Each training sequence is one svara occurrence (recording_id, svara_ann_idx).
Segments are ordered by seg_idx and include CP, STA, TR.

Encoder input per segment (ENC_INPUT_DIM = 11):
    seg_type_oh(6) | dur_rel(1) | delta_norm(1) | log_k_raw(1) | logit_s_raw(1) | A_raw(1)
    For CP: last 3 dims = 0 (no tanh+osc fit).

Decoder input per segment (DEC_INPUT_DIM = 9):
    seg_type_oh(6) | dur_rel(1) | delta_norm(1) | dy0_required_norm(1)
    dy0_required_norm = tanh(dy0_required / M_SCALE)
    dy0_required = norm_deriv_at(0, k_gt, s_gt, A_gt) for STA/TR; 0 otherwise.
    At inference: propagated from m1 of the previous STA/TR segment.

Targets (OUTPUT_DIM = 3), masked to STA/TR only:
    log_k_raw = log(k) / LOG_K_SCALE
    logit_s_raw = logit(s) / LOGIT_S_SCALE
    A_raw = A / A_SCALE

Load:
    from src.models.param_gru.dataset_param_gru import build_dataset, collate_param_batch
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import settings as S

# ── types ─────────────────────────────────────────────────────────────────────

TYPE_TO_IDX = {"CP": 0, "SIL": 1, "STAp": 2, "STAt": 3, "TRa": 4, "TRd": 5}
_N_TYPES    = 6

# ── normalization constants (match model_param_gru.py) ───────────────────────

K_MIN, K_MAX   = 0.3, 10.0
S_MIN, S_MAX   = 0.01, 0.99
A_SCALE        = 0.4        # A_osc in [-0.4, 0.2]; scale by max(|A_MIN|, A_MAX)
LOG_K_SCALE    = math.log(K_MAX)           # ≈ 2.30
LOGIT_S_SCALE  = 5.0                       # logit([0.01,0.99]) ≈ [-4.6, 4.6]
SLOPE_SCALE    = 300.0                     # ¢/s; tanh squash (kept for diagnostics)
M_SCALE        = 5.0                       # normalized slope (dimensionless); tanh squash

ENC_INPUT_DIM  = 11  # seg_oh(6) + dur_rel + delta_norm + log_k_raw + logit_s_raw + A_raw
DEC_INPUT_DIM  = 9   # seg_oh(6) + dur_rel + delta_norm + dy0_required_norm
OUTPUT_DIM     = 3   # log_k_raw, logit_s_raw, A_raw

# Svara labels (alphabetical, matches gruvae)
from src.models.gruvae.dataset_gruvae import SVARA_TO_IDX, SVARA_LABELS  # noqa: E402

SHAPES_PATH = S.INTERIM_ANALYSIS / "segment_shapes.parquet"

from src.models.curve_vae.fit_sta_tr_curves import norm_deriv_at  # noqa: E402


# ── boundary slope helpers ────────────────────────────────────────────────────

def _slope_end_np(seg_type: str, k: float, s: float, A: float,
                  concavity: float, delta: float, dur: float) -> float:
    """Slope at end of segment in cents/s (numpy, not differentiable)."""
    dur = max(dur, 1e-6)
    if seg_type in ("STAp", "STAt", "TRa", "TRd"):
        h1 = np.tanh(k * (1 - s))
        h0 = np.tanh(-k * s)
        denom = h1 - h0
        if abs(denom) < 1e-9:
            return delta / dur
        dh1 = k / np.cosh(k * (1 - s)) ** 2 - 2 * math.pi * A
        return (dh1 / denom) * delta / dur
    elif seg_type == "CP":
        a = 0.0 if (concavity is None or not math.isfinite(concavity)) else concavity
        b = (delta - a * dur ** 2) / dur
        return 2 * a * dur + b
    return 0.0  # SIL


def _compute_slopes(rows: list[dict]) -> list[float]:
    """slope_end for every segment in the sequence (in cents/s)."""
    result = []
    for r in rows:
        delta = (r["end_cents"] or 0.0) - (r["start_cents"] or 0.0)
        result.append(_slope_end_np(
            r["seg_type"],
            r["k_steep"]   if r["k_steep"]   is not None else 1.0,
            r["s_inflect"] if r["s_inflect"] is not None else 0.5,
            r["A_osc"]     if r["A_osc"]     is not None else 0.0,
            r["concavity"] if r["concavity"] is not None else 0.0,
            delta, r["dur_sec"],
        ))
    return result


# ── per-segment feature builders ──────────────────────────────────────────────

def _safe_log_k(k: float | None) -> float:
    v = k if (k is not None and math.isfinite(k) and k > 0) else 1.0
    return math.log(max(v, 1e-6)) / LOG_K_SCALE


def _safe_logit_s(s: float | None) -> float:
    v = s if (s is not None and math.isfinite(s) and 0 < s < 1) else 0.5
    v = max(S_MIN, min(S_MAX, v))
    return math.log(v / (1 - v)) / LOGIT_S_SCALE


def _safe_A(A: float | None) -> float:
    v = A if (A is not None and math.isfinite(A)) else 0.0
    return v / A_SCALE


# ── dataset ───────────────────────────────────────────────────────────────────

class ParamSequenceDataset(Dataset):
    """Each item is one svara occurrence (variable-length sequence of segments)."""

    def __init__(self, sequences: list[dict]):
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        return self.sequences[idx]


def _build_sequence(rows: list[dict], total_dur: float, svara_idx: int) -> dict | None:
    """Build one sequence dict from sorted segment rows. Returns None if unusable."""
    n = len(rows)
    if n == 0:
        return None

    slope_ends = _compute_slopes(rows)

    enc_input       = np.zeros((n, ENC_INPUT_DIM), dtype=np.float32)
    dec_struct      = np.zeros((n, DEC_INPUT_DIM - 1), dtype=np.float32)  # without dy0_required
    dy0_required_gt = np.zeros(n, dtype=np.float32)  # GT required start slope per step
    targets         = np.zeros((n, OUTPUT_DIM), dtype=np.float32)
    target_mask     = np.zeros(n, dtype=bool)
    dur_sec_arr     = np.zeros(n, dtype=np.float32)
    delta_cents_arr = np.zeros(n, dtype=np.float32)

    for i, r in enumerate(rows):
        seg   = r["seg_type"]
        oh    = [0.0] * _N_TYPES
        oh[TYPE_TO_IDX.get(seg, 0)] = 1.0

        delta     = (r["end_cents"] or 0.0) - (r["start_cents"] or 0.0)
        dur_rel   = r["dur_sec"] / total_dur if total_dur > 1e-6 else 0.0
        delta_n   = delta / 1200.0

        # encoder input: structural + params (zeros for non-STA/TR)
        log_k  = _safe_log_k(r.get("k_steep"))
        lgts   = _safe_logit_s(r.get("s_inflect"))
        A_n    = _safe_A(r.get("A_osc"))
        has_params = seg in ("STAp", "STAt", "TRa", "TRd") and r.get("k_steep") is not None
        enc_input[i] = oh + [dur_rel, delta_n,
                             log_k if has_params else 0.0,
                             lgts  if has_params else 0.0,
                             A_n   if has_params else 0.0]

        # decoder structural (slope_prev added in collate)
        dec_struct[i] = oh + [dur_rel, delta_n]

        dur_sec_arr[i]    = r["dur_sec"]
        delta_cents_arr[i] = delta

        # dy0_required for this step: the normalized slope that the curve
        # at this step should naturally start with (from its own GT fit).
        # For STA/TR with fitted params: norm_deriv_at(0, k, s, A).
        # For CP/SIL (or missing params): 0 — no constraint.
        if has_params:
            k_v = r["k_steep"]  if r["k_steep"]  is not None else 1.0
            s_v = r["s_inflect"] if r["s_inflect"] is not None else 0.5
            A_v = r["A_osc"]    if r["A_osc"]    is not None else 0.0
            dy0_required_gt[i] = norm_deriv_at(k_v, s_v, A_v, at_end=False)

        # target
        if has_params:
            targets[i]     = [_safe_log_k(r["k_steep"]),
                               _safe_logit_s(r["s_inflect"]),
                               _safe_A(r["A_osc"])]
            target_mask[i] = True

    # decoder input: append dy0_required_norm (tanh-squashed normalized slope)
    dy0_required_norm = np.tanh(dy0_required_gt / M_SCALE).reshape(n, 1)
    dec_input = np.concatenate([dec_struct, dy0_required_norm], axis=1).astype(np.float32)

    return {
        "enc_input":         torch.from_numpy(enc_input),
        "dec_input":         torch.from_numpy(dec_input),
        "targets":           torch.from_numpy(targets),
        "target_mask":       torch.from_numpy(target_mask),
        "dy0_required_norm": torch.from_numpy(dy0_required_norm.squeeze(1).astype(np.float32)),
        "slope_end_gt":      torch.tensor(slope_ends, dtype=torch.float32),
        "dur_sec":           torch.from_numpy(dur_sec_arr),
        "delta_cents":       torch.from_numpy(delta_cents_arr),
        "svara_idx":    torch.tensor(svara_idx, dtype=torch.long),
        "total_dur":    torch.tensor(total_dur, dtype=torch.float32),
        "length":       torch.tensor(n, dtype=torch.long),
    }


def build_dataset(path: Path = SHAPES_PATH) -> ParamSequenceDataset:
    df = pl.read_parquet(path)
    sequences = []

    for (rec, ann_idx), group in df.group_by(["recording_id", "svara_ann_idx"]):
        group = group.sort("seg_idx")
        rows  = group.to_dicts()

        svara_label = rows[0]["svara_label"]
        if svara_label not in SVARA_TO_IDX:
            continue
        svara_idx  = SVARA_TO_IDX[svara_label]
        total_dur  = sum(r["dur_sec"] for r in rows)

        seq = _build_sequence(rows, total_dur, svara_idx)
        if seq is not None:
            sequences.append(seq)

    return ParamSequenceDataset(sequences)


# ── collate ───────────────────────────────────────────────────────────────────

def collate_param_batch(batch: list[dict]) -> dict:
    lengths  = torch.tensor([item["length"] for item in batch], dtype=torch.long)
    T        = int(lengths.max().item())
    B        = len(batch)

    enc_input        = torch.zeros(B, T, ENC_INPUT_DIM)
    dec_input        = torch.zeros(B, T, DEC_INPUT_DIM)
    targets          = torch.zeros(B, T, OUTPUT_DIM)
    target_mask      = torch.zeros(B, T, dtype=torch.bool)
    dy0_required_norm = torch.zeros(B, T)
    slope_end_gt     = torch.zeros(B, T)
    dur_sec          = torch.zeros(B, T)
    delta_cents      = torch.zeros(B, T)
    svara_idx        = torch.zeros(B, dtype=torch.long)
    total_dur        = torch.zeros(B)

    for i, item in enumerate(batch):
        n = item["length"].item()
        enc_input[i, :n]         = item["enc_input"]
        dec_input[i, :n]         = item["dec_input"]
        targets[i, :n]           = item["targets"]
        target_mask[i, :n]       = item["target_mask"]
        dy0_required_norm[i, :n] = item["dy0_required_norm"]
        slope_end_gt[i, :n]      = item["slope_end_gt"]
        dur_sec[i, :n]           = item["dur_sec"]
        delta_cents[i, :n]       = item["delta_cents"]
        svara_idx[i]             = item["svara_idx"]
        total_dur[i]             = item["total_dur"]

    return {
        "enc_input":         enc_input,
        "dec_input":         dec_input,
        "targets":           targets,
        "target_mask":       target_mask,
        "dy0_required_norm": dy0_required_norm,
        "slope_end_gt":      slope_end_gt,
        "dur_sec":           dur_sec,
        "delta_cents":       delta_cents,
        "svara_idx":         svara_idx,
        "total_dur":         total_dur,
        "lengths":           lengths,
    }
