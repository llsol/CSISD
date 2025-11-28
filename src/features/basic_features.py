import numpy as np

def compute_base_features(pitch_cents, time_sec):

    mask = np.isfinite(pitch_cents)
    if not np.any(mask):
        return {
            "duration_sec": np.nan,
            "pitch_mean": np.nan,
            "pitch_std": np.nan,
            "pitch_range": np.nan,
            "slope_global": np.nan,
        }



    p = pitch_cents[mask]
    t = time_sec[mask]

    duration_sec = float(t[-1] - t[0])

    pitch_mean  = float(np.mean(p))
    pitch_std   = float(np.std(p))
    pitch_range = float(np.max(p) - np.min(p))

    if duration_sec > 0:
        slope_global = float((p[-1] - p[0]) / duration_sec)
    else:
        slope_global = np.nan

    return {
        "duration_sec": duration_sec,
        "pitch_mean": pitch_mean,
        "pitch_std": pitch_std,
        "pitch_range": pitch_range,
        "slope_global": slope_global,
    }
    return df