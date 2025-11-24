import polars as pl
import numpy as np
from scipy.interpolate import CubicSpline

def interpolate_gaps(df: pl.DataFrame, max_gap_sec: float) -> pl.DataFrame:
    """
    Interpolate small gaps in pitch data up to max_gap_sec.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame with 'time_sec' and 'pitch_hz' columns.
    max_gap_sec : float
        Maximum gap duration to interpolate.

    Returns
    -------
    pl.DataFrame
        DataFrame with interpolated pitch values.
    """

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
        pl.Series("f0_interpolated", f_interp)
    ])

    return df

def compute_valid_regions(df: pl.DataFrame) -> pl.DataFrame:

    df = df.with_columns(pl.col("f0_interpolated").is_not_null().alias("valid_for_spline"))
    df = df.with_columns(((pl.col("valid_for_spline") != pl.col("valid_for_spline").shift(1)).cast(pl.Int32).cumsum()).alias("group_id"))
    
    return df





def fit_splines_to_groups(
        df: pl.DataFrame,
        min_points: int = 7,
        min_duration: float = 0.050  # en segons
) -> pl.DataFrame:
    
    t = df["time_rel_sec"].to_numpy()
    y = df["f0_interpolated"].to_numpy()
    group = df["group_id"].to_numpy()

    f_spline = np.full_like(y, np.nan, dtype=float)

    unique_groups = np.unique(group)

    for g in unique_groups:
        mask = (group == g)
        idx = np.where(mask)[0]

        if len(idx) < min_points:
            continue

        t_group = t[idx]
        y_group = y[idx]

        duration = t_group[-1] - t_group[0]
        if duration < min_duration:
            continue
    
        try:
            spline = CubicSpline(t_group, y_group, bc_type="natural", extrapolate=False)
        except Exception:
            continue
    
        f_spline[idx] = spline(t_group)

    df = df.with_columns(pl.Series("f0_spline", f_spline))

    return df
