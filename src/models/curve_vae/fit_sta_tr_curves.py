"""
Fit parametric models to normalized GT pitch curves.

Unified 3-parameter model for STA and TR:

    h(t; k, s, A) = tanh(k·(t − s))  +  A·sin(2π·(t − 0.5))
    y(t)          = [h(t) − h(0)] / [h(1) − h(0)]

    k > 0      : steepness of the sigmoid transition
    s ∈ (0,1)  : skew — location of the inflection point
    A ∈ ℝ      : oscillation amplitude
                   A =  0 → pure tanh (monotone)
                   A <  0 (moderate) → plateau / slow-down at centre
                   A << 0 → N-shape (derivative goes negative at centre)
                   A >  0 → W-shape of derivative (boosted tails)

    STA and TR: y(t)   (0 → 1; synthesis scales to actual cents range)

Endpoints are always exactly 0 and 1 regardless of A:
    sin(2π·(0−0.5)) = sin(−π) = 0   ✓
    sin(2π·(1−0.5)) = sin( π) = 0   ✓

Outputs a parquet with the original curve columns plus:
    k_steep, s_inflect, A_osc, rmse, r2

Usage:
    python -m src.models.curve_vae.fit_curves
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import settings as S

DEFAULT_CURVES = S.CURVE_VAE_DIR / "gt_curves.parquet"
DEFAULT_OUT    = S.CURVE_VAE_DIR / "gt_curves_fitted.parquet"

K_MIN, K_MAX = 0.3,  10.0
S_MIN, S_MAX = 0.01, 0.99
A_MIN, A_MAX = -0.4,  0.2


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _h(t: np.ndarray, k: float, s: float, A: float) -> np.ndarray:
    return np.tanh(k * (t - s)) + A * np.sin(2 * np.pi * (t - 0.5))


def curve_model(t: np.ndarray, k: float, s: float, A: float) -> np.ndarray:
    """STA: normalized tanh+oscillation, 0 → 1."""
    t  = np.clip(t, 0.0, 1.0)
    h  = _h(t, k, s, A)
    lo = _h(np.float64(0.0), k, s, A)
    hi = _h(np.float64(1.0), k, s, A)
    d  = hi - lo
    return (h - lo) / d if abs(d) > 1e-12 else t.copy()


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0


def fit_curve(t: np.ndarray, p: np.ndarray) -> tuple[float, float, float, float, float]:
    """Multi-start fit. Returns (k, s, A, rmse, r2)."""
    fn = curve_model

    best: tuple[float, float, float, float] | None = None  # (k, s, A, rmse)
    for k0 in [1.0, 3.0, 8.0]:
        for s0 in [0.2, 0.5, 0.8]:
            for A0 in [0.0, -0.3]:
                try:
                    popt, _ = curve_fit(
                        fn, t, p,
                        p0=[k0, s0, A0],
                        bounds=([K_MIN, S_MIN, A_MIN], [K_MAX, S_MAX, A_MAX]),
                        maxfev=4000,
                    )
                    y_hat = fn(t, *popt)
                    rmse  = float(np.sqrt(np.mean((p - y_hat) ** 2)))
                    if best is None or rmse < best[3]:
                        best = (float(popt[0]), float(popt[1]), float(popt[2]), rmse)
                except Exception:
                    pass

    if best is None:
        best = (1.0, 0.5, 0.0, float("nan"))

    k, s, A, _ = best
    y_hat = fn(t, k, s, A)
    rmse  = float(np.sqrt(np.mean((p - y_hat) ** 2)))
    r2    = r_squared(p, y_hat)
    return k, s, A, rmse, r2


# ---------------------------------------------------------------------------
# Corpus fitting with progress
# ---------------------------------------------------------------------------

def fit_corpus(df: pl.DataFrame) -> pl.DataFrame:
    n      = len(df)
    ks, ss, As, rmses, r2s = [], [], [], [], []
    t0     = time.time()
    report = max(1, n // 20)   # print every ~5 %

    rows = df.iter_rows(named=True)
    for i, row in enumerate(rows):
        t = np.array(row["t_norm"],  dtype=np.float64)
        p = np.array(row["p_norm"],  dtype=np.float64)
        k, s, A, rmse, r2 = fit_curve(t, p)
        ks.append(k); ss.append(s); As.append(A)
        rmses.append(rmse); r2s.append(r2)

        if (i + 1) % report == 0 or i == n - 1:
            pct     = (i + 1) / n * 100
            elapsed = time.time() - t0
            eta     = elapsed / (i + 1) * (n - i - 1)
            print(f"  [{i+1:>5}/{n}]  {pct:5.1f}%  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s  "
                  f"last rmse={rmse:.4f}")

    return df.with_columns([
        pl.Series("k_steep",   ks,    dtype=pl.Float32),
        pl.Series("s_inflect", ss,    dtype=pl.Float32),
        pl.Series("A_osc",     As,    dtype=pl.Float32),
        pl.Series("rmse",      rmses, dtype=pl.Float32),
        pl.Series("r2",        r2s,   dtype=pl.Float32),
    ])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curves", default=str(DEFAULT_CURVES))
    parser.add_argument("--out",    default=str(DEFAULT_OUT))
    args = parser.parse_args()

    df = pl.read_parquet(args.curves)
    print(f"Fitting tanh+osc model to {len(df)} curves...")

    df_fit = fit_corpus(df)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df_fit.write_parquet(out)
    print(f"\nSaved → {out}")

    for seg_type in ("STAp", "STAt", "TRa", "TRd"):
        sub = df_fit.filter(pl.col("seg_type") == seg_type)
        print(f"\n{seg_type}  (n={len(sub)})")
        if len(sub) == 0:
            continue
        print(f"  k:    median={sub['k_steep'].median():.3f}  "
              f"p5={sub['k_steep'].quantile(0.05):.3f}  "
              f"p95={sub['k_steep'].quantile(0.95):.3f}")
        print(f"  s:    median={sub['s_inflect'].median():.3f}  "
              f"p5={sub['s_inflect'].quantile(0.05):.3f}  "
              f"p95={sub['s_inflect'].quantile(0.95):.3f}")
        print(f"  A:    median={sub['A_osc'].median():.4f}  "
              f"p5={sub['A_osc'].quantile(0.05):.4f}  "
              f"p95={sub['A_osc'].quantile(0.95):.4f}")
        n_nshape = int((sub["A_osc"] < -0.15).sum())
        print(f"  N-shape (A<−0.15): {n_nshape}  ({100*n_nshape/len(sub):.1f}%)")
        print(f"  R²:   median={sub['r2'].median():.4f}")
        print(f"  RMSE: median={sub['rmse'].median():.4f}")


if __name__ == "__main__":
    main()
