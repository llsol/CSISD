
import numpy as np


def compute_derivatives(pitch_cents, time_sec):
    
    mask = np.isfinite(pitch_cents)
    if np.sum(mask) < 3:
        nan_arr = np.full_like(pitch_cents, np.nan)
        return {"deriv1": nan_arr, "deriv2": nan_arr}

    p = pitch_cents
    t = time_sec


    dt1 = np.diff(t)
    dp1 = np.diff(p)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = dp1 / dt1

    deriv1 = np.concatenate(([np.nan], d1))


    dt2 = t[2:] - t[:-2]
    dd1 = np.diff(d1)
    with np.errstate(divide="ignore", invalid="ignore"):
        d2 = dd1 / dt2

    deriv2 = np.concatenate(([np.nan, np.nan], d2))

    return {"deriv1": deriv1, "deriv2": deriv2}


def compute_derivative_features(deriv1, deriv2):
   
    def stats(x):
        x = x[np.isfinite(x)]
        if len(x) == 0:
            return (np.nan, np.nan, np.nan, np.nan)
        return (
            float(np.mean(x)),
            float(np.std(x)),
            float(np.mean(np.abs(x))),
            float(np.std(np.abs(x))),
        )

    d1_mean, d1_std, d1_abs_mean, d1_abs_std = stats(deriv1)
    d2_mean, d2_std, d2_abs_mean, d2_abs_std = stats(deriv2)

    return {
        "deriv1_mean": d1_mean,
        "deriv1_std": d1_std,
        "deriv1_abs_mean": d1_abs_mean,
        "deriv1_abs_std": d1_abs_std,
        "deriv2_mean": d2_mean,
        "deriv2_std": d2_std,
        "deriv2_abs_mean": d2_abs_mean,
        "deriv2_abs_std": d2_abs_std,
    }
