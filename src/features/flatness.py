import numpy as np
import polars as pl


def extract_flat_regions(
    df: pl.DataFrame,
    *,
    time_col: str = "time_rel_sec",
    pitch_col: str = "f0_savgol_p3_w13",
    candidate_col: str = "flat_candidate",
    out_col: str = "flat_region",
    min_duration_sec: float = 0.150,
    cent_tolerance: float = 10.0,          # tolerance in cents (works for both cents + Hz)
    pitch_unit: str = "cents",             # "cents" or "hz"
    # --- NEW: candidate from abs(deriv1) ---
    d1_threshold: float = 1000.0,          # cents/sec
    abs_deriv1_col: str | None = None,     # e.g. "abs_deriv1_cents_per_sec" to store it
    verbose: bool = False,
) -> pl.DataFrame:
    """
    Mark non-overlapping maximal 'flat' regions inside runs where candidate_col == True.

    If candidate_col is missing, it is created as:
        abs(deriv1_cents_per_sec) < d1_threshold

    Deriv1 is computed in cents/sec:
      - pitch_unit == "cents": dp = (cents[i] - cents[i-1])
      - pitch_unit == "hz":    dp = 1200*log2(hz[i]/hz[i-1])
      and then dp/dt.
    """

    if pitch_unit not in ("cents", "hz"):
        raise ValueError("pitch_unit must be 'cents' or 'hz'")

    df0 = df.with_row_index("row_nr")

    # -------------------------------------------------
    # 0) CREATE candidate_col from abs(deriv1) if missing
    # -------------------------------------------------
    if candidate_col not in df0.columns:
        t = df0[time_col].to_numpy()
        p = df0[pitch_col].to_numpy()

        # dt
        dt = np.empty_like(t, dtype=float)
        dt[:] = np.nan
        dt[1:] = t[1:] - t[:-1]

        # dp in cents
        dp_cents = np.empty_like(p, dtype=float)
        dp_cents[:] = np.nan

        if pitch_unit == "cents":
            dp_cents[1:] = p[1:] - p[:-1]
        else:
            # hz -> cents interval
            # dp = 1200*log2(f[i]/f[i-1])
            valid = np.isfinite(p[1:]) & np.isfinite(p[:-1]) & (p[1:] > 0) & (p[:-1] > 0)
            tmp = np.full(p[1:].shape, np.nan, dtype=float)
            tmp[valid] = 1200.0 * np.log2(p[1:][valid] / p[:-1][valid])
            dp_cents[1:] = tmp

        # deriv (cents/sec)
        deriv1 = np.full_like(p, np.nan, dtype=float)
        valid_d = np.isfinite(dp_cents) & np.isfinite(dt) & (dt > 0)
        deriv1[valid_d] = dp_cents[valid_d] / dt[valid_d]

        abs_deriv1 = np.abs(deriv1)
        cand_arr = (abs_deriv1 < float(d1_threshold)) & np.isfinite(abs_deriv1)

        # add columns
        if abs_deriv1_col is not None:
            df0 = df0.with_columns(pl.Series(abs_deriv1_col, abs_deriv1))
        df0 = df0.with_columns(pl.Series(candidate_col, cand_arr))

    # -------------------------------------------------
    # 1) Build runs of consecutive candidates
    # -------------------------------------------------
    cand_expr = pl.col(candidate_col).fill_null(False).cast(pl.Boolean)
    df0 = df0.with_columns(cand_expr.alias("_cand"))

    df0 = df0.with_columns(
        (pl.col("_cand") != pl.col("_cand").shift(1))
        .cast(pl.Int32)
        .fill_null(1)
        .cum_sum()
        .alias("run_id")
    )

    n_total = df0.height
    stable_mask = np.zeros(n_total, dtype=bool)

    runs = (
        df0.filter(pl.col("_cand") == True)
           .select("run_id")
           .unique()
           .sort("run_id")
           .to_series()
           .to_list()
    )

    def range_in_cents(win_min: float, win_max: float) -> float:
        if pitch_unit == "cents":
            return win_max - win_min
        if win_min <= 0 or not np.isfinite(win_min) or not np.isfinite(win_max):
            return np.inf
        return 1200.0 * np.log2(win_max / win_min)

    # -------------------------------------------------
    # 2) Maximal non-overlapping stable segments
    # -------------------------------------------------
    for rid in runs:
        sub = (
            df0.filter(pl.col("run_id") == rid)
               .select(["row_nr", time_col, pitch_col])
               .sort("row_nr")
        )

        row_nrs = sub["row_nr"].to_numpy()
        times = sub[time_col].to_numpy()
        pitches = sub[pitch_col].to_numpy()

        valid = np.isfinite(times) & np.isfinite(pitches)
        if pitch_unit == "hz":
            valid &= (pitches > 0)

        row_nrs = row_nrs[valid]
        times = times[valid]
        pitches = pitches[valid]

        m = len(row_nrs)
        if m == 0:
            continue

        i = 0
        while i < m:
            j = i
            while j < m and (times[j] - times[i]) < min_duration_sec:
                j += 1
            if j >= m:
                break

            win_min = float(np.min(pitches[i:j + 1]))
            win_max = float(np.max(pitches[i:j + 1]))
            r = range_in_cents(win_min, win_max)

            if r > 2.0 * cent_tolerance:
                i += 1
                continue

            j_max = j
            k = j + 1
            while k < m:
                p = float(pitches[k])
                if p < win_min:
                    win_min = p
                if p > win_max:
                    win_max = p

                r = range_in_cents(win_min, win_max)
                if r <= 2.0 * cent_tolerance:
                    j_max = k
                    k += 1
                else:
                    break

            stable_mask[row_nrs[i:j_max + 1]] = True
            if verbose:
                print(
                    f"[✔] {times[i]:.3f}-{times[j_max]:.3f}s | "
                    f"range≈{range_in_cents(win_min, win_max):.2f} cents | "
                    f"n={j_max - i + 1}"
                )
            i = j_max + 1

    out = (
        df0.with_columns(pl.Series(out_col, stable_mask))
           .drop(["row_nr", "run_id", "_cand"])
    )
    return out


#def extract_flat_regions(
#    df: pl.DataFrame,
#    *,
#    time_col: str = "time_rel_sec",
#    pitch_col: str = "f0_savgol_p3_w13",
#    candidate_col: str = "flat_candidate",
#    out_col: str = "flat_region",
#    min_duration_sec: float = 0.150,
#    cent_tolerance: float = 10.0,          # tolerance in cents (works for both cents + Hz)
#    pitch_unit: str = "cents",             # "cents" or "hz"
#    verbose: bool = False,
#) -> pl.DataFrame:
#    """
#    Mark non-overlapping maximal 'flat' regions inside runs where candidate_col == True.
#
#    - Non-overlapping segmentation: once a stable segment [i..j] is found, next start is j+1.
#    - Works with pitch in cents or Hz:
#        * if pitch_unit == "cents": range_cents = max(window) - min(window)
#        * if pitch_unit == "hz":    range_cents = 1200*log2(max(window)/min(window))
#      (so cent_tolerance is always interpreted in cents)
#    """
#
#    if pitch_unit not in ("cents", "hz"):
#        raise ValueError("pitch_unit must be 'cents' or 'hz'")
#
#    df0 = df.with_row_index("row_nr")
#
#    # Make candidate boolean and build run_id for consecutive stretches
#    cand = pl.col(candidate_col).fill_null(False).cast(pl.Boolean)
#    df0 = df0.with_columns(
#    cand.alias("_cand")
#)
#
#    df0 = df0.with_columns(
#        (pl.col("_cand") != pl.col("_cand").shift(1))
#        .cast(pl.Int32)
#        .fill_null(1)
#        .cum_sum()
#        .alias("run_id")
#    )
#
#
#    n_total = df0.height
#    stable_mask = np.zeros(n_total, dtype=bool)
#
#    # Only run_ids where candidate is True
#    runs = (
#        df0.filter(pl.col("_cand") == True)
#           .select("run_id")
#           .unique()
#           .sort("run_id")
#           .to_series()
#           .to_list()
#    )
#
#    # Helper: window "range" expressed in cents (even if input is Hz)
#    def range_in_cents(win_min: float, win_max: float) -> float:
#        if pitch_unit == "cents":
#            return win_max - win_min
#        # hz
#        if win_min <= 0 or not np.isfinite(win_min) or not np.isfinite(win_max):
#            return np.inf
#        return 1200.0 * np.log2(win_max / win_min)
#
#    # Iterate each candidate run
#    for rid in runs:
#        sub = (
#            df0.filter(pl.col("run_id") == rid)
#               .select(["row_nr", time_col, pitch_col])
#               .sort("row_nr")
#        )
#
#        row_nrs = sub["row_nr"].to_numpy()
#        times = sub[time_col].to_numpy()
#        pitches = sub[pitch_col].to_numpy()
#
#        # clean pitch values inside the run (keep alignment via mask)
#        valid = np.isfinite(times) & np.isfinite(pitches)
#        if pitch_unit == "hz":
#            valid &= (pitches > 0)
#        row_nrs = row_nrs[valid]
#        times = times[valid]
#        pitches = pitches[valid]
#
#        m = len(row_nrs)
#        if m == 0:
#            continue
#
#        i = 0
#        while i < m:
#            # 1) find the smallest j that satisfies min_duration
#            j = i
#            while j < m and (times[j] - times[i]) < min_duration_sec:
#                j += 1
#            if j >= m:
#                break  # no segment from i can reach min duration
#
#            # 2) try to build a maximal segment starting at i
#            win_min = float(np.min(pitches[i:j + 1]))
#            win_max = float(np.max(pitches[i:j + 1]))
#            r = range_in_cents(win_min, win_max)
#
#            if r > 2.0 * cent_tolerance:
#                # can't start a stable segment at i; move forward
#                i += 1
#                continue
#
#            # 3) expand to the right as far as the constraint holds
#            j_max = j
#            k = j + 1
#            while k < m:
#                p = float(pitches[k])
#                if p < win_min:
#                    win_min = p
#                if p > win_max:
#                    win_max = p
#
#                r = range_in_cents(win_min, win_max)
#                if r <= 2.0 * cent_tolerance:
#                    j_max = k
#                    k += 1
#                else:
#                    break
#
#            # 4) mark and jump (non-overlapping)
#            stable_mask[row_nrs[i:j_max + 1]] = True
#            if verbose:
#                print(
#                    f"[✔] {times[i]:.3f}-{times[j_max]:.3f}s | "
#                    f"range≈{range_in_cents(win_min, win_max):.2f} cents | "
#                    f"n={j_max - i + 1}"
#                )
#            i = j_max + 1
#
#    out = (
#        df0.with_columns(pl.Series(out_col, stable_mask))
#           .drop(["row_nr", "run_id", "_cand"])
#    )
#    return out



#def extract_flat_regions(
#    df: pl.DataFrame,
#    time_col: str = "time_rel_sec",
#    pitch_col: str = "f0_savgol_p3_w13",
#    candidate_col: str = "flat_candidate",
#    out_col: str = "flat_region",
#    min_duration_sec: float = 0.150,
#    apply_pitch_range_check: bool = True,
#    cent_tolerance: float = 10.0,  
#    verbose: bool = False,
#) -> pl.DataFrame:
#
#    # Afegim un índex de fila explícit
#    df0 = df.with_row_index("row_nr")
#
#    # Construïm run_id per trams consecutius de candidate_col
#    df0 = df0.with_columns([
#        (pl.col(candidate_col) != pl.col(candidate_col).shift(1))
#        .cast(pl.Int32)
#        .fill_null(1)
#        .cum_sum()
#        .alias("run_id")
#    ])
#
#    # Total files
#    n_total = df0.height
#    stable_mask = np.zeros(n_total, dtype=bool)
#
#    # Ens quedem només runs on candidate_col és True
#    runs = (
#        df0.filter(pl.col(candidate_col) == True)
#           .select(["run_id"])
#           .unique()
#           .sort("run_id")
#           .to_series()
#           .to_list()
#    )
#
#    # Iterem per cada run
#    for rid in runs:
#        sub = df0.filter(pl.col("run_id") == rid).select(["row_nr", time_col, pitch_col])
#
#        row_nrs = sub["row_nr"].to_numpy()
#        times = sub[time_col].to_numpy()
#        pitches = sub[pitch_col].to_numpy()
#
#        m = len(row_nrs)
#        i = 0
#        while i < m:
#            j = i + 1
#            while j < m and (times[j] - times[i]) < min_duration_sec:
#                j += 1
#
#            while j < m:
#                window = pitches[i:j+1]
#                if apply_pitch_range_check:
#                    pitch_range = np.nanmax(window) - np.nanmin(window)
#                    ok = pitch_range <= 2.0 * cent_tolerance
#                else:
#                    pitch_range = 0.0
#                    ok = True
#
#                if ok:
#                    stable_mask[row_nrs[i:j+1]] = True
#                    if verbose:
#                        print(f"[✔] Subregion {times[i]:.3f}-{times[j]:.3f}s | range={pitch_range:.1f} cents")
#                    i = j + 1
#                    break
#                else:
#                    j -= 1
#            else:
#                i += 1
#
#    # Afegim la columna al DF original (sense mantenir columnes auxiliars)
#    out = (
#        df0.with_columns(pl.Series(out_col, stable_mask))
#           .drop(["row_nr", "run_id"])
#    )
#    return out
#