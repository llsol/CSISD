"""
Conditional parametric curve generative model for STA and TR segments.

Model family (3 params):
    h(t; k, s, A) = tanh(k·(t−s)) + A·sin(2π·(t−0.5))
    y(t)          = [h(t)−h(0)] / [h(1)−h(0)]

    STA: y(t)          — normally 0→1
    TR:  1 − y(t)      — normally 1→0

    A=0          → monotone tanh (pure S-curve)
    A moderate < 0 → plateau / slow-down at centre
    A very < 0   → N-shape (derivative goes negative at centre)

Distribution:
    [log(k), logit(s), A] ~ N(μ[svara], Σ[svara])   per seg_type × svara
    (unconstrained reparameterisation keeps k>0, s∈(0,1), A free)

Usage:
    python -m src.models.curve_vae.curve_model
"""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import settings as S
from src.models.curve_vae.fit_sta_tr_curves import curve_model, curve_model_tr

FITTED       = S.DATA_INTERIM / "models" / "curve_vae" / "gt_curves_fitted.parquet"
SVARA_LABELS = sorted(['D', 'G', 'M', 'N', 'P', 'R', 'S'])
MIN_COUNT    = 15


# ---------------------------------------------------------------------------
# Unconstrained transforms
# ---------------------------------------------------------------------------

def _to_free(k: np.ndarray, s: np.ndarray, A: np.ndarray) -> np.ndarray:
    """(k, s, A) → [log(k), logit(s), A]  unconstrained ℝ³."""
    log_k  = np.log(k.clip(1e-4))
    logit_s = np.log(s.clip(1e-6) / (1 - s).clip(1e-6))
    return np.column_stack([log_k, logit_s, A])


def _from_free(z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """[log(k), logit(s), A] → (k, s, A)."""
    k = np.exp(z[:, 0])
    s = 1.0 / (1.0 + np.exp(-z[:, 1]))
    A = z[:, 2]
    return k, s, A


# ---------------------------------------------------------------------------
# Distribution
# ---------------------------------------------------------------------------

@dataclass
class KSADist:
    """Trivariate Gaussian over [log(k), logit(s), A]."""
    mean: np.ndarray   # (3,)
    cov:  np.ndarray   # (3, 3)

    def sample(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        z = rng.multivariate_normal(self.mean, self.cov, size=n)
        return _from_free(z)


# ---------------------------------------------------------------------------
# Main model class
# ---------------------------------------------------------------------------

@dataclass
class CurveModel:
    dists: dict[str, dict[str, KSADist]] = field(default_factory=dict)

    def fit(self, df: pl.DataFrame) -> "CurveModel":
        for seg_type in ("STA", "TR"):
            self.dists[seg_type] = {}
            sub = df.filter(pl.col("seg_type") == seg_type)
            self._fit_one(sub, seg_type, "pooled")
            for sv in SVARA_LABELS:
                sv_sub = sub.filter(pl.col("svara_label") == sv)
                if len(sv_sub) >= MIN_COUNT:
                    self._fit_one(sv_sub, seg_type, sv)
        return self

    def _fit_one(self, df: pl.DataFrame, seg_type: str, key: str) -> None:
        k = df["k_steep"].to_numpy().astype(np.float64).clip(1e-4)
        s = df["s_inflect"].to_numpy().astype(np.float64).clip(1e-6, 1 - 1e-6)
        A = df["A_osc"].to_numpy().astype(np.float64)
        Z    = _to_free(k, s, A)
        mean = Z.mean(axis=0)
        cov  = np.cov(Z.T) + np.eye(3) * 1e-6
        self.dists[seg_type][key] = KSADist(mean=mean, cov=cov)

    def _dist(self, seg_type: str, svara: str | None) -> KSADist:
        d = self.dists[seg_type]
        if svara and svara in d:
            return d[svara]
        return d["pooled"]

    def generate(
        self,
        seg_type: str,
        svara: str | None = None,
        n: int = 1,
        rng: np.random.Generator | None = None,
    ) -> list[dict]:
        """
        Generate n parametric curves.

        Returns list of dicts:
            k, s, A, curve_fn: Callable[[np.ndarray], np.ndarray],
            is_nshape: bool  (A < −0.15)
        """
        if rng is None:
            rng = np.random.default_rng()

        fn  = curve_model if seg_type == "STA" else curve_model_tr
        ks, ss, As = self._dist(seg_type, svara).sample(n, rng)

        return [
            {
                "seg_type":  seg_type,
                "k":         float(k),
                "s":         float(s),
                "A":         float(A),
                "is_nshape": float(A) < -0.15,
                "curve_fn":  (lambda t, _k=k, _s=s, _A=A: fn(t, _k, _s, _A)),
            }
            for k, s, A in zip(ks, ss, As)
        ]

    def summary(self) -> str:
        lines = ["CurveModel — tanh+osc(k, s, A) trivariate Gaussian:"]
        for seg_type in ("STA", "TR"):
            for key in ["pooled"] + SVARA_LABELS:
                if key in self.dists.get(seg_type, {}):
                    d     = self.dists[seg_type][key]
                    k_med = float(np.exp(d.mean[0]))
                    s_med = float(1 / (1 + np.exp(-d.mean[1])))
                    A_med = float(d.mean[2])
                    lines.append(
                        f"  {seg_type} [{key:>6}]  "
                        f"k={k_med:.2f}  s={s_med:.3f}  A={A_med:+.4f}"
                    )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    df    = pl.read_parquet(FITTED)
    model = CurveModel().fit(df)
    print(model.summary())

    rng = np.random.default_rng(0)
    t   = np.linspace(0, 1, 50)

    print("\nSTA (svara=G, 4 samples):")
    for g in model.generate("STA", svara="G", n=4, rng=rng):
        y    = g["curve_fn"](t)
        tag  = " [N-shape]" if g["is_nshape"] else ""
        print(f"  k={g['k']:.2f}  s={g['s']:.3f}  A={g['A']:+.3f}{tag}  "
              f"y_min={y.min():.3f}  y_max={y.max():.3f}")

    print("\nTR (pooled, 4 samples):")
    for g in model.generate("TR", n=4, rng=rng):
        y    = g["curve_fn"](t)
        tag  = " [N-shape]" if g["is_nshape"] else ""
        print(f"  k={g['k']:.2f}  s={g['s']:.3f}  A={g['A']:+.3f}{tag}  "
              f"y_min={y.min():.3f}  y_max={y.max():.3f}")


if __name__ == "__main__":
    main()
