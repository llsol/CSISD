import numpy as np
import polars as pl

def compute_base_features(
    df: pl.DataFrame,
    pitch_column: str = "f0_savgol_p3_w13",
    time_column: str = "time_rel_sec",
):
    """
    Calcula features bàsiques d'un segment a partir d'un Polars DataFrame.
    Retorna un dict d'escalars.
    """
    pitch_cents = df[pitch_column].to_numpy()
    time_sec = df[time_column].to_numpy()

    mask = np.isfinite(pitch_cents)
    if not np.any(mask):
        return {
            "duration_sec": np.nan,
            "pitch_mean": np.nan,
            "pitch_median": np.nan,
            "pitch_std": np.nan,
            "pitch_range": np.nan,
            "slope_global": np.nan,
        }

    p = pitch_cents[mask]
    t = time_sec[mask]

    duration_sec = float(t[-1] - t[0])

    pitch_mean = float(np.mean(p))
    pitch_median = float(np.median(p))
    pitch_std = float(np.std(p))
    pitch_range = float(np.max(p) - np.min(p))

    if duration_sec > 0:
        slope_global = float((p[-1] - p[0]) / duration_sec)
    else:
        slope_global = np.nan

    return {
        "duration_sec": duration_sec,
        "pitch_mean": pitch_mean,
        "pitch_median": pitch_median,
        "pitch_std": pitch_std,
        "pitch_range": pitch_range,
        "slope_global": slope_global,
    }