
import numpy as np
from scipy.signal import savgol_filter
import polars as pl


def compute_derivatives(
        df: pl.DataFrame,
        col_in: str = "f0_pchip",
        window_length: int = 13,
        polyorder: int = 3,
):


    f = df[col_in].to_numpy()
    group = df["group_id"].to_numpy()

    groups = np.unique(group)

    d1_savgol = np.full_like(f, np.nan, dtype=float)
    d2_savgol = np.full_like(f, np.nan, dtype=float)

    for g in groups:
        mask = (group == g)
        idx = np.where(mask)[0]

        f_group = f[idx]

        if len(idx) < window_length:
            continue
        
        not_nan = np.isfinite(f_group)
        if not np.any(not_nan):
            continue

        first = np.argmax(not_nan)
        last = len(f_group) - 1 - np.argmax(not_nan[::-1])

        f_trim = f_group[first:last+1]

        if np.any(np.isnan(f_trim)):
            continue

        if len(f_trim) < window_length:
            continue

        try:
            f_trim_d1 = savgol_filter(f_trim, window_length=window_length, polyorder=polyorder, deriv=1, mode='interp')
            f_trim_d2 = savgol_filter(f_trim, window_length=window_length, polyorder=polyorder, deriv=2, mode='interp')
        except Exception:
            print(f"Could not apply Savitzky-Golay filter to group {g}.")
            continue

        f_group_d1 = np.full_like(f_group, np.nan, dtype=float)
        f_group_d2 = np.full_like(f_group, np.nan, dtype=float)
        f_group_d1[first:last+1] = f_trim_d1
        f_group_d2[first:last+1] = f_trim_d2

        d1_savgol[idx] = f_group_d1
        d2_savgol[idx] = f_group_d2

    return {"deriv1": d1_savgol, "deriv2": d2_savgol}


def compute_derivative_features(deriv1, deriv2):
   
    def stats(x):
        x = x[np.isfinite(x)]
        if len(x) == 0:
            return (np.nan, np.nan, np.nan, np.nan)
        return (
            float(np.mean(x)),
            float(np.std(x)),
            float(np.mean(np.abs(x))),
            float(np.std(np.abs(x)))
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
