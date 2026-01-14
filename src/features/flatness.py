import numpy as np
import polars as pl

def extract_flat_regions(
    df: pl.DataFrame,
    time_col: str = "time_rel_sec",
    pitch_col: str = "f0_savgol_p3_w13",
    candidate_col: str = "flat_candidate",
    out_col: str = "flat_region",
    min_duration_sec: float = 0.150,
    apply_pitch_range_check: bool = True,
    cent_tolerance: float = 10.0,  
    verbose: bool = False,
) -> pl.DataFrame:

    # Afegim un índex de fila explícit
    df0 = df.with_row_index("row_nr")

    # Construïm run_id per trams consecutius de candidate_col
    df0 = df0.with_columns([
        (pl.col(candidate_col) != pl.col(candidate_col).shift(1))
        .cast(pl.Int32)
        .fill_null(1)
        .cum_sum()
        .alias("run_id")
    ])

    # Total files
    n_total = df0.height
    stable_mask = np.zeros(n_total, dtype=bool)

    # Ens quedem només runs on candidate_col és True
    runs = (
        df0.filter(pl.col(candidate_col) == True)
           .select(["run_id"])
           .unique()
           .sort("run_id")
           .to_series()
           .to_list()
    )

    # Iterem per cada run
    for rid in runs:
        sub = df0.filter(pl.col("run_id") == rid).select(["row_nr", time_col, pitch_col])

        row_nrs = sub["row_nr"].to_numpy()
        times = sub[time_col].to_numpy()
        pitches = sub[pitch_col].to_numpy()

        m = len(row_nrs)
        i = 0
        while i < m:
            j = i + 1
            while j < m and (times[j] - times[i]) < min_duration_sec:
                j += 1

            while j < m:
                window = pitches[i:j+1]
                if apply_pitch_range_check:
                    pitch_range = np.nanmax(window) - np.nanmin(window)
                    ok = pitch_range <= 2.0 * cent_tolerance
                else:
                    pitch_range = 0.0
                    ok = True

                if ok:
                    stable_mask[row_nrs[i:j+1]] = True
                    if verbose:
                        print(f"[✔] Subregion {times[i]:.3f}-{times[j]:.3f}s | range={pitch_range:.1f} cents")
                    i = j + 1
                    break
                else:
                    j -= 1
            else:
                i += 1

    # Afegim la columna al DF original (sense mantenir columnes auxiliars)
    out = (
        df0.with_columns(pl.Series(out_col, stable_mask))
           .drop(["row_nr", "run_id"])
    )
    return out
