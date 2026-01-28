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





def add_flat_id(
    df: pl.DataFrame,
    *,
    flat_col: str = "flat_region",
    out_col: str = "flat_id",
) -> pl.DataFrame:
    """
    Add a run-based id for consecutive flat regions.

    - out_col is null where flat_col is False/null
    - out_col increments for each new consecutive True run
    """
    if flat_col not in df.columns:
        raise ValueError(f"Missing '{flat_col}'")

    flat = pl.col(flat_col).fill_null(False).cast(pl.Boolean)

    # run change marker over flat boolean
    run_change = (flat != flat.shift(1)).fill_null(True).cast(pl.Int32).cum_sum()

    # keep ids only where flat is True
    return df.with_columns(
        pl.when(flat)
          .then(run_change)
          .otherwise(None)
          .alias(out_col)
    )

