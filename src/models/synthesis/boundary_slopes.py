"""
Boundary slope initialization for synthesis.

At svara segment boundaries the slope chain has no natural predecessor/successor:
  - Start of svara, or STA/TR right after SIL  → Case B (start boundary)
  - TR right before SIL or at end of svara      → Case A (end boundary)

Case B — dy0_required for a STA/TR at start/after SIL:
  Generates a virtual predecessor (k,s,A) from CurveModel for the same seg_type;
  samples (delta_v, dur_v) from GT distribution; converts the predecessor's m1 to
  the normalized start slope the boundary segment should have.

Case A — m1_required for a TR before SIL/end:
  If a successor segment is known (after SIL): use its actual (delta, dur) with a
  virtual (k,s,A) drawn from CurveModel for its seg_type.
  If no successor (TR at end of svara): sample m1 directly from GT distribution.
  After computing m1_required, A is adjusted analytically to satisfy it while
  preserving k and s (= shape character of the curve).
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.models.curve_vae.fit_sta_tr_curves import norm_deriv_at, A_MIN, A_MAX
from src.models.curve_vae.sta_tr_model import CurveModel

import settings as S

_STA_TR   = {"STAp", "STAt", "TRa", "TRd"}
_MIN_DELTA = 1.0   # cents — below this, slope is ill-defined
_MIN_DUR   = 1e-4  # seconds


# ── analytic A adjustment ──────────────────────────────────────────────────────

def solve_A_for_m1(k: float, s: float, m1_required: float) -> float:
    """Solve analytically for A so that norm'(1, k, s, A) = m1_required.

    norm'(1) = [k/cosh²(k(1-s)) − 2πA] / d   (d = tanh(k(1-s)) + tanh(ks))
    → A = [k/cosh²(k(1-s)) − m1_required · d] / (2π)

    Returns A clamped to [A_MIN, A_MAX].
    d is independent of A (sin vanishes at endpoints), so the formula is exact.
    """
    h0  = math.tanh(-k * s)
    h1  = math.tanh(k * (1.0 - s))
    d   = h1 - h0
    if abs(d) < 1e-9:
        return 0.0
    dh1 = k / math.cosh(k * (1.0 - s)) ** 2
    A   = (dh1 - m1_required * d) / (2.0 * math.pi)
    return float(np.clip(A, A_MIN, A_MAX))


# ── sampler ────────────────────────────────────────────────────────────────────

@dataclass
class BoundarySlopeSampler:
    """Samples boundary slopes from GT distributions.

    Parameters
    ----------
    df_shapes : DataFrame from segment_shapes.parquet
        Needs: seg_type, svara_label, start_cents, end_cents, dur_sec.
    curve_model : fitted CurveModel
        Used to generate virtual (k,s,A) for virtual predecessor/successor.
    """

    df_shapes:   pl.DataFrame
    curve_model: CurveModel

    # ── internal helpers ───────────────────────────────────────────────────────

    def _sample_row(
        self,
        seg_type: str,
        svara: str,
        rng: np.random.Generator,
    ) -> dict | None:
        sub = self.df_shapes.filter(
            (pl.col("seg_type") == seg_type) & (pl.col("svara_label") == svara)
        )
        if len(sub) < 5:
            sub = self.df_shapes.filter(pl.col("seg_type") == seg_type)
        if len(sub) == 0:
            return None
        return sub.row(int(rng.integers(len(sub))), named=True)

    def _virtual_ksa(
        self,
        seg_type: str,
        svara: str,
        rng: np.random.Generator,
    ) -> tuple[float, float, float]:
        gen = self.curve_model.generate(seg_type=seg_type, svara=svara, n=1, rng=rng)[0]
        return gen["k"], gen["s"], gen["A"]

    # ── Case B: start boundary ─────────────────────────────────────────────────

    def sample_dy0_for_boundary_start(
        self,
        seg_type: str,
        svara: str,
        delta_boundary: float,
        dur_boundary: float,
        rng: np.random.Generator,
    ) -> float:
        """dy0_required (normalized) for a STA/TR at the start of a svara or after SIL.

        Generates a virtual predecessor of *seg_type*, samples its (delta, dur)
        from the GT distribution, and converts its m1 to the normalized start slope
        the boundary segment should have.

        Returns 0.0 for degenerate inputs (delta ≈ 0, or no GT data).
        """
        if abs(delta_boundary) < _MIN_DELTA or dur_boundary < _MIN_DUR:
            return 0.0

        k_v, s_v, A_v = self._virtual_ksa(seg_type, svara, rng)
        row = self._sample_row(seg_type, svara, rng)
        if row is None:
            return 0.0

        delta_v = (row.get("end_cents") or 0.0) - (row.get("start_cents") or 0.0)
        dur_v   = float(row.get("dur_sec") or 0.3)
        if abs(delta_v) < _MIN_DELTA or dur_v < _MIN_DUR:
            return 0.0

        m1_v    = norm_deriv_at(k_v, s_v, A_v, at_end=True)
        v_end_v = m1_v * delta_v / dur_v           # physical slope ¢/s
        return float(v_end_v * dur_boundary / delta_boundary)

    # ── Case A: end boundary ───────────────────────────────────────────────────

    def compute_m1_from_successor(
        self,
        seg_type_succ: str,
        svara: str,
        delta_succ: float,
        dur_succ: float,
        delta_TR: float,
        dur_TR: float,
        rng: np.random.Generator,
    ) -> float | None:
        """m1_required (normalized) for a TR before SIL, given the successor segment.

        Generates virtual (k,s,A) for the successor, computes its start slope,
        and converts to the TR's required end slope.
        Returns None for degenerate inputs.
        """
        if abs(delta_succ) < _MIN_DELTA or dur_succ < _MIN_DUR:
            return None
        if abs(delta_TR) < _MIN_DELTA or dur_TR < _MIN_DUR:
            return None

        k_v, s_v, A_v = self._virtual_ksa(seg_type_succ, svara, rng)
        m0_v      = norm_deriv_at(k_v, s_v, A_v, at_end=False)
        v_start_v = m0_v * delta_succ / dur_succ   # physical slope ¢/s
        return float(v_start_v * dur_TR / delta_TR)

    def sample_m1_for_boundary_end(
        self,
        seg_type: str,
        svara: str,
        rng: np.random.Generator,
    ) -> float:
        """m1_required (normalized) for a TR at the very end of a svara (no successor).

        Samples (k,s,A) from CurveModel and returns its implied m1.
        """
        k_v, s_v, A_v = self._virtual_ksa(seg_type, svara, rng)
        return norm_deriv_at(k_v, s_v, A_v, at_end=True)


# ── factory ────────────────────────────────────────────────────────────────────

def load_boundary_sampler(curve_model: CurveModel) -> BoundarySlopeSampler:
    """Load BoundarySlopeSampler from segment_shapes.parquet."""
    shapes_path = S.INTERIM_ANALYSIS / "segment_shapes.parquet"
    df = pl.read_parquet(shapes_path).filter(
        pl.col("seg_type").is_in(list(_STA_TR))
    )
    return BoundarySlopeSampler(df_shapes=df, curve_model=curve_model)
