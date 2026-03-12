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
    max_velocity_sts: float = 260.0,
    dev_thresh_cents: float = 600.0,   # es manté per compatibilitat, però només escala alguns llindars
    expand_neighbors: int = 1,
    f_ref=None
) -> pl.DataFrame:
    """
    Detect outliers and overwrite `col_f0` with NaN at outlier positions.

        1- detect local jumps
        2- detect short runs that are incoherent with the context using:
                - context excursion
                - context match
                - shape (std, range, curvature, sign changes)
                - edge jumps

    """

    def _find_true_runs(mask_bool: np.ndarray):
        runs = []
        in_run = False
        start = None

        for i, v in enumerate(mask_bool):
            if v and not in_run:
                start = i
                in_run = True
            elif not v and in_run:
                runs.append((start, i))  # end exclusiu
                in_run = False

        if in_run:
            runs.append((start, len(mask_bool)))

        return runs

    def _fill_bridged_short_gaps(mask_bool: np.ndarray, max_gap: int) -> np.ndarray:
        """
        If there are two outlier points/runs separated by a short gap, fill the gap.
        """
        if max_gap <= 0:
            return mask_bool

        mask = mask_bool.copy()
        runs = _find_true_runs(mask)

        if len(runs) < 2:
            return mask

        for (s1, e1), (s2, e2) in zip(runs[:-1], runs[1:]):
            gap = s2 - e1
            if 0 < gap <= max_gap:
                mask[e1:s2] = True

        return mask

    def _count_invalid_left(mask_valid: np.ndarray, idx_start: int) -> int:
        c = 0
        j = idx_start - 1
        while j >= 0 and not mask_valid[j]:
            c += 1
            j -= 1
        return c

    def _count_invalid_right(mask_valid: np.ndarray, idx_end_exclusive: int) -> int:
        c = 0
        j = idx_end_exclusive
        while j < len(mask_valid) and not mask_valid[j]:
            c += 1
            j += 1
        return c

    def _shape_stats(y: np.ndarray):
        """
        Geometric stats of the run.
        """
        n = len(y)

        if n == 0:
            return {
                "std": 0.0,
                "range": 0.0,
                "mean_d2": 0.0,
                "n_sign_changes": 0,
                "peak_pos_norm": 0.5,
                "trough_pos_norm": 0.5,
            }

        y = np.asarray(y, dtype=float)

        run_std = float(np.std(y)) if n > 1 else 0.0
        run_range = float(np.max(y) - np.min(y)) if n > 1 else 0.0

        if n < 3:
            return {
                "std": run_std,
                "range": run_range,
                "mean_d2": 0.0,
                "n_sign_changes": 0,
                "peak_pos_norm": 0.5,
                "trough_pos_norm": 0.5,
            }

        d1 = np.diff(y)
        d1_nz = d1[np.abs(d1) > 1e-9]

        if len(d1_nz) <= 1:
            n_sign_changes = 0
        else:
            signs = np.sign(d1_nz)
            n_sign_changes = int(np.sum(signs[1:] != signs[:-1]))

        d2 = np.diff(d1)
        mean_d2 = float(np.mean(d2)) if len(d2) > 0 else 0.0

        peak_pos = int(np.argmax(y))
        trough_pos = int(np.argmin(y))
        denom = max(1, n - 1)

        return {
            "std": run_std,
            "range": run_range,
            "mean_d2": mean_d2,
            "n_sign_changes": n_sign_changes,
            "peak_pos_norm": peak_pos / denom,
            "trough_pos_norm": trough_pos / denom,
        }

    def _detect_context_incoherent_runs(
        f_cents_full: np.ndarray,
        mask_valid_full: np.ndarray,
        is_outlier_seed_full: np.ndarray,
        max_run_len: int = 12,
        context_len: int = 4,
        excursion_thresh_cents: float = 180.0,
        context_match_thresh_cents: float = 140.0,
        max_internal_std_cents: float = 120.0,
        max_internal_range_cents: float = 260.0,
        max_slope_sign_changes: int = 1,
        max_edge_jump_cents_per_sample: float = 140.0,
        gap_bonus_invalid: int = 1,
    ) -> np.ndarray:
        """
        Detect short runs that are incoherent with the context using:
        """
        N = len(f_cents_full)
        out = np.zeros(N, dtype=bool)

        valid_runs = _find_true_runs(mask_valid_full)

        for s, e in valid_runs:
            if e <= s:
                continue

            for rs in range(s, e):
                for re in range(rs + 1, min(e, rs + max_run_len) + 1):
                    y = f_cents_full[rs:re]
                    if len(y) == 0 or not np.all(np.isfinite(y)):
                        continue

                    left_ctx = f_cents_full[max(s, rs - context_len):rs]
                    right_ctx = f_cents_full[re:min(e, re + context_len)]

                    left_ctx = left_ctx[np.isfinite(left_ctx)]
                    right_ctx = right_ctx[np.isfinite(right_ctx)]

                    if len(left_ctx) < 2 or len(right_ctx) < 2:
                        continue

                    med_left = float(np.median(left_ctx))
                    med_mid = float(np.median(y))
                    med_right = float(np.median(right_ctx))

                    excursion_left = abs(med_mid - med_left)
                    excursion_right = abs(med_mid - med_right)
                    context_match = abs(med_left - med_right)

                    # el centre ha d'estar separat del context i el context ha de ser coherent
                    if not (
                        excursion_left >= excursion_thresh_cents
                        and excursion_right >= excursion_thresh_cents
                        and context_match <= context_match_thresh_cents
                    ):
                        continue

                    stats = _shape_stats(y)

                    context_center = 0.5 * (med_left + med_right)
                    is_up_island = med_mid > context_center
                    is_down_island = med_mid < context_center

                    # salts d'entrada/sortida
                    jump_in = abs(y[0] - med_left)
                    jump_out = abs(y[-1] - med_right)

                    # normalització discreta: si hi ha mostres invàlides entremig, tolerem més canvi
                    n_invalid_left = _count_invalid_left(mask_valid_full, rs)
                    n_invalid_right = _count_invalid_right(mask_valid_full, re)

                    jump_in_per_sample = jump_in / max(1, 1 + n_invalid_left)
                    jump_out_per_sample = jump_out / max(1, 1 + n_invalid_right)

                    touches_gap = (n_invalid_left >= gap_bonus_invalid) or (n_invalid_right >= gap_bonus_invalid)

                    # llavor procedent del detector de salts
                    seed_nearby = False
                    if rs > 0 and is_outlier_seed_full[rs - 1]:
                        seed_nearby = True
                    if re < N and is_outlier_seed_full[re]:
                        seed_nearby = True
                    if np.any(is_outlier_seed_full[rs:re]):
                        seed_nearby = True

                    shape_bad = False

                    # massa pla per la seva distància al context
                    if (
                        stats["std"] <= max_internal_std_cents
                        and stats["range"] <= max_internal_range_cents
                    ):
                        shape_bad = True

                    # massa canvis de signe a la derivada
                    if stats["n_sign_changes"] > max_slope_sign_changes:
                        shape_bad = True

                    # illa cap amunt + curvatura còncava -> sospitosa
                    if is_up_island and stats["mean_d2"] > 0:
                        shape_bad = True

                    # illa cap avall + curvatura convexa -> sospitosa
                    if is_down_island and stats["mean_d2"] < 0:
                        shape_bad = True

                    # pic o vall massa lateral
                    if is_up_island and (stats["peak_pos_norm"] < 0.15 or stats["peak_pos_norm"] > 0.85):
                        shape_bad = True
                    if is_down_island and (stats["trough_pos_norm"] < 0.15 or stats["trough_pos_norm"] > 0.85):
                        shape_bad = True

                    edge_bad = (
                        jump_in_per_sample > max_edge_jump_cents_per_sample
                        or jump_out_per_sample > max_edge_jump_cents_per_sample
                    )

                    gap_bad = False
                    if touches_gap:
                        if (
                            stats["std"] <= (max_internal_std_cents * 1.35)
                            and context_match <= (context_match_thresh_cents * 0.9)
                        ):
                            gap_bad = True

                    if shape_bad or edge_bad or gap_bad or seed_nearby:
                        out[rs:re] = True

        return out


    t = df[col_time].to_numpy()
    f = df[col_f0].to_numpy()
    too_long = df[col_too_long].to_numpy()

    N = len(f)

    mask_valid = (f > 0) & (~too_long)
    valid_idx = np.where(mask_valid)[0]

    if len(valid_idx) < 3:
        return df.with_columns([
            pl.Series("is_outlier", np.zeros(N, dtype=bool)),
            pl.Series(col_f0, f.copy()),
        ])

    if f_ref is None:
        f_ref = np.nanmedian(f[mask_valid])

    f_cents = np.full(N, np.nan, dtype=float)
    f_cents[mask_valid] = 1200.0 * np.log2(f[mask_valid] / f_ref)


    max_vel_cents = max_velocity_sts * 100.0

    dt_all = np.diff(t[np.isfinite(t)])
    dt_median = np.median(dt_all) if len(dt_all) > 0 else 0.01
    if not np.isfinite(dt_median) or dt_median <= 0:
        dt_median = 0.01

    max_jump_cents_per_sample = max_vel_cents * dt_median

    is_outlier_vel = np.zeros(N, dtype=bool)

    prev_idx = valid_idx[0]
    prev_cents = f_cents[prev_idx]
    prev_time = t[prev_idx]

    for idx in valid_idx[1:]:
        curr_cents = f_cents[idx]
        curr_time = t[idx]

        dt = curr_time - prev_time
        sample_gap = idx - prev_idx

        if dt <= 0 or sample_gap <= 0:
            prev_idx = idx
            prev_cents = curr_cents
            prev_time = curr_time
            continue

        dcents = abs(curr_cents - prev_cents)

        max_d_allowed_time = max_vel_cents * dt
        max_d_allowed_samples = max_jump_cents_per_sample * sample_gap

        max_d_allowed = max(max_d_allowed_time, max_d_allowed_samples)

        if dcents > max_d_allowed:
            is_outlier_vel[idx] = True

        prev_idx = idx
        prev_cents = curr_cents
        prev_time = curr_time

    is_outlier_vel = _fill_bridged_short_gaps(is_outlier_vel, max_gap=6)


    is_outlier_shape = _detect_context_incoherent_runs(
        f_cents_full=f_cents,
        mask_valid_full=mask_valid,
        is_outlier_seed_full=is_outlier_vel,
        max_run_len=12,
        context_len=4,
        excursion_thresh_cents=max(160.0, 0.45 * dev_thresh_cents),
        context_match_thresh_cents=140.0,
        max_internal_std_cents=120.0,
        max_internal_range_cents=260.0,
        max_slope_sign_changes=1,
        max_edge_jump_cents_per_sample=140.0,
        gap_bonus_invalid=1,
    )

    # =========================================================
    # combinació final
    # =========================================================

    is_outlier = is_outlier_vel | is_outlier_shape

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
'''

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

'''

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