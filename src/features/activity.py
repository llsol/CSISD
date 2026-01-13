import numpy as np
from .derivatives import compute_derivatives

def contour_activity(
        deriv1,
        normalize=False
        ):
    d1 = np.asarray(deriv1, float)

    abs_d1 = np.abs(d1)

    contour = np.nancumsum(abs_d1)

    total = np.nansum(abs_d1)
    mean = np.nanmean(abs_d1)

    if normalize and total > 0:
        contour = contour / total

    return {
        "contour_activity": contour,
        "activity_sum": total,
        "activity_mean": mean,
    }



def _moving_average(x, window_size):
    if window_size is None or window_size <= 1:
        return x

    x = np.asarray(x, float)
    n = len(x)
    y = np.full_like(x, np.nan, dtype=float)

    mask = np.isfinite(x)
    if not np.any(mask):
        return y
    
    kernel = np.ones(window_size, dtype=float)
    valid = np.convolve(mask.astype(float), kernel, mode="same")
    smoothed = np.convolve(np.nan_to_num(x, nan=0.0), kernel, mode="same")

    with np.errstate(invalid="ignore", divide="ignore"):
        y = smoothed / valid

    y[valid == 0] = np.nan

    return y





def compute_activity(pitch_cents, time_sec, alpha=0.5, smooth_window=None):
    
    derivs = compute_derivatives(pitch_cents, time_sec)
    d1 = derivs["deriv1"]
    d2 = derivs["deriv2"]

    # activity scalar
    with np.errstate(invalid="ignore"):
        activity = np.abs(d1) + alpha * np.abs(d2)

    if smooth_window is not None and smooth_window > 1:
        activity = _moving_average(activity, smooth_window)

    return activity, d1, d2





def compute_activity_features(activity):

    a = np.asarray(activity, float)
    a = a[np.isfinite(a)]

    if len(a) == 0:
        return {
            "activity_mean": np.nan,
            "activity_std": np.nan,
            "activity_max": np.nan,
            "activity_p90": np.nan,
        }

    activity_mean = float(np.mean(a))
    activity_std  = float(np.std(a))
    activity_max  = float(np.max(a))
    activity_p90  = float(np.percentile(a, 90.0))

    return {
        "activity_mean": activity_mean,
        "activity_std": activity_std,
        "activity_max": activity_max,
        "activity_p90": activity_p90,
    }





def compute_activity_threshold(activity, quantile=0.2):

    a = np.asarray(activity, float)
    a = a[np.isfinite(a)]

    if len(a) == 0:
        return np.nan
    
    q = float(np.clip(quantile, 0.0, 1.0))

    return float(np.percentile(a, q * 100.0))





def compute_activity_mask(activity, threshold):
    
    a = np.asarray(activity, float)
    
    with np.errstate(invalid="ignore"):
        mask = a > threshold

    mask = np.where(np.isfinite(a), mask, False)

    return mask