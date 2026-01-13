import polars as pl
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter
from scipy.signal import medfilt


def detect_zero_runs(df: pl.DataFrame, max_gap_sec: float) -> pl.DataFrame:

    if "f0_Hz" not in df.columns or "time_rel_sec" not in df.columns:
        raise ValueError("df must contain 'f0_Hz' and 'time_rel_sec'.")

    t = df["time_rel_sec"].to_numpy()
    f = df["f0_Hz"].to_numpy()

    N = len(df)

    is_zero = (f == 0)

    prev_zero = np.concatenate(([False], is_zero[:-1]))
    next_zero = np.concatenate((is_zero[1:], [False]))

    gap_starts = (is_zero == True) & (prev_zero == False)
    gap_ends = (is_zero == True) & (next_zero == False)

    start_idx = np.where(gap_starts)[0]
    end_idx = np.where(gap_ends)[0]

    if len(start_idx) == 0:
        return (
            np.array([], dtype=int),
            np.array([], dtype=int),
            np.array([], dtype=float)
        )
    
    durations = t[end_idx] - t[start_idx]


    return start_idx, end_idx, durations





def mark_gap_regions(
        df: pl.DataFrame,
        start_indices: np.ndarray,
        end_indices: np.ndarray,
        durations: np.ndarray,
        max_gap_sec: float
) -> pl.DataFrame:
    
    f = df["f0_Hz"].to_numpy()
    N = len(f)

    too_long_mask = np.zeros(N, dtype=bool)
    f_interp = f.astype(float).copy()  

    for start, end, dur in zip(start_indices, end_indices, durations):

        if dur > max_gap_sec:
            too_long_mask[start:end+1] = True
        else:
            f_interp[start:end+1] = np.nan

    df = df.with_columns([
        pl.Series("too_long_to_interp", too_long_mask),
        pl.Series("f0_interpolated", f_interp),

    ])

    return df





def remove_outliers(
    df: pl.DataFrame,
    col_time: str = "time_rel_sec",
    col_f0: str = "f0_interpolated",
    col_too_long: str = "too_long_to_interp",
    max_velocity_sts: float = 200.0,
    dev_thresh_cents: float = 600.0,
    expand_neighbors: int = 1,
    f_ref = None
) -> pl.DataFrame:
    
    t = df[col_time].to_numpy()
    f = df[col_f0].to_numpy()

    too_long = df[col_too_long].to_numpy()

    N = len(f)

    mask_valid = (f > 0) & (~too_long)
    valid_idx = np.where(mask_valid)[0]

    if len(valid_idx) < 3:
        return df.with_columns(pl.Series("is_outlier", np.zeros(N, dtype=bool)))
    
    if f_ref is None:
        f_ref = np.nanmedian(f[mask_valid])

    f_cents = np.full(N, np.nan, dtype=float)
    f_cents[mask_valid] = 1200.0 * np.log2(f[mask_valid] / f_ref)

    max_vel_cents = max_velocity_sts * 100.0

    is_outlier_vel = np.zeros(N, dtype=bool)

    prev_idx = valid_idx[0]
    prev_cents = f_cents[prev_idx]
    prev_time = t[prev_idx]

    for idx in valid_idx[1:]:
        curr_cents = f_cents[idx]
        curr_time = t[idx]

    
        dt = curr_time - prev_time
        if dt <= 0:
            prev_idx = idx
            prev_cents = curr_cents
            prev_time = curr_time
            continue

        dcents = abs(curr_cents - prev_cents)

        max_d_allowed = max_vel_cents * dt

        if np.abs(dcents) > max_d_allowed:
            is_outlier_vel[idx] = True

        prev_idx = idx
        prev_cents = curr_cents
        prev_time = curr_time

    f_cents_valid = f_cents[valid_idx]

    kernel_size = 31
    if kernel_size > len(f_cents_valid):
        kernel_size = len(f_cents_valid) if len(f_cents_valid) % 2 == 1 else len(f_cents_valid) - 1
    if kernel_size < 3:
        kernel_size = 3

    median_local = medfilt(f_cents_valid, kernel_size=kernel_size)

    dev = np.abs(f_cents_valid - median_local)
    is_outlier_dev_local = dev > dev_thresh_cents

    is_outlier_dev = np.zeros(N, dtype=bool)
    is_outlier_dev[valid_idx] = is_outlier_dev_local

    is_outlier = is_outlier_vel | is_outlier_dev

    if expand_neighbors > 0:
        is_outlier_expanded = is_outlier.copy()
        for shift in range(1, expand_neighbors + 1):
            is_outlier_expanded[:-shift] |= is_outlier[shift:]
            is_outlier_expanded[shift:] |= is_outlier[:-shift]
        is_outlier = is_outlier_expanded

    is_outlier &= mask_valid

    f_clean = f.copy()
    f_clean[is_outlier] = np.nan

    df = df.with_columns([
        pl.Series("is_outlier", is_outlier),
        pl.Series(col_f0, f_clean),
    ])

    return df




def compute_valid_regions(df: pl.DataFrame) -> pl.DataFrame:

    df = df.with_columns(
        (
            (pl.col("f0_interpolated") > 0)
        ).alias("valid_for_pchip")    
    )

    df = df.with_columns(
        (pl.col("valid_for_pchip") != pl.col("valid_for_pchip").shift(1))
        .cast(pl.Int32)
        .alias("change_flag")
    )    
    df = df.with_columns(
        pl.col("change_flag").cum_sum().alias("group_id")
    )
    df = df.drop("change_flag")
    return df





def fit_pchip_to_groups(
        df: pl.DataFrame,
        pchip_min_points: int = 7,
        pchip_min_duration: float = 0.050  # en segons
) -> pl.DataFrame:
    
    t = df["time_rel_sec"].to_numpy()
    y = df["f0_interpolated"].to_numpy()
    group = df["group_id"].to_numpy()

    f_pchip = np.full_like(y, np.nan, dtype=float)

    unique_groups = np.unique(group)

    for g in unique_groups:
        mask = (group == g)
        idx = np.where(mask)[0]

        if len(idx) < pchip_min_points:
            continue

        t_group = t[idx]
        y_group = y[idx]

        duration = t_group[-1] - t_group[0]
        if duration < pchip_min_duration:
            continue

        mask_valid = np.isfinite(y_group) & (y_group > 0)
        if mask_valid.sum() < pchip_min_points:
            continue

        t_valid = t_group[mask_valid]
        y_valid = y_group[mask_valid]

        try:
            interp = PchipInterpolator(t_valid, y_valid, extrapolate=False)
        except Exception:
            continue
    
        f_pchip[idx] = interp(t_group)

    df = df.with_columns(pl.Series("f0_pchip", f_pchip))

    return df





def apply_savgol_filter(
        df: pl.DataFrame,
        col_in: str = "f0_pchip",
        window_length: int = 13,
        polyorder: int = 3,
        col_out = "f0_savgol",
) -> pl.DataFrame:

    col_out = f"{col_out}_p{polyorder}_w{window_length}"

    f = df[col_in].to_numpy()
    group = df["group_id"].to_numpy()

    groups = np.unique(group)

    f_savgol = np.full_like(f, np.nan, dtype=float)

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
            f_trim_smoothed = savgol_filter(f_trim, window_length=window_length, polyorder=polyorder, mode='interp')
        except Exception:
            print(f"Could not apply Savitzky-Golay filter to group {g}.")
            continue

        f_group_smooth = f_group.copy()
        f_group_smooth[first:last+1] = f_trim_smoothed

        f_savgol[idx] = f_group_smooth

    df = df.with_columns(pl.Series(col_out, f_savgol))

    return df





def preprocess_pitch(
        df: pl.DataFrame,
        max_gap_sec: float = 0.100,
        pchip_min_points: int = 7,
        pchip_min_duration: float = 0.050,
        apply_savgol_13_3: bool = True,
        apply_savgol_27_3: bool = True,
        savgol_window_length: int = None,
        savgol_polyorder: int = None,

) -> pl.DataFrame:
    
    start_idx, end_idx, durations = detect_zero_runs(df, max_gap_sec)

    df = mark_gap_regions(df, start_idx, end_idx, durations, max_gap_sec)

    df = remove_outliers(df)

    df = compute_valid_regions(df)

    df = fit_pchip_to_groups(df, pchip_min_points, pchip_min_duration)

    if apply_savgol_13_3:
        df = apply_savgol_filter(
            df,
            col_in="f0_pchip",
            window_length=13,
            polyorder=3,
            col_out="f0_savgol"
        )

    if apply_savgol_27_3:
        df = apply_savgol_filter(
            df,
            col_in="f0_pchip",
            window_length=27,
            polyorder=3,
            col_out="f0_savgol"
        )
    
    if savgol_window_length is not None and savgol_polyorder is not None:
        df = apply_savgol_filter(
            df,
            col_in="f0_pchip",
            window_length=savgol_window_length,
            polyorder=savgol_polyorder,
            col_out="f0_savgol"
        )

    return df