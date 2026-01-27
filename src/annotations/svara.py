import polars as pl
import numpy as np
import polars as pl
import numpy as np


def attach_svara_annotations_to_pitch(
    df_pitch: pl.DataFrame,
    df_svaras: pl.DataFrame,
) -> pl.DataFrame:
    df_pitch = df_pitch.with_row_index("row_nr")

    t = df_pitch.get_column("time_rel_sec").to_numpy()
    n = df_pitch.height

    # --- 1) índex més proper (nearest) per start/end ---
    # fem-ho en numpy per simplicitat/claritat (df_svaras sol ser petit)
    start_times = df_svaras.get_column("start_time_sec").to_numpy()
    end_times = df_svaras.get_column("end_time_sec").to_numpy()

    start_idx = np.array([int(np.abs(t - s).argmin()) for s in start_times], dtype=np.int64)
    end_idx = np.array([int(np.abs(t - e).argmin()) for e in end_times], dtype=np.int64)

    df_svaras = df_svaras.with_columns([
        pl.Series("_start_row_nr", start_idx),
        pl.Series("_end_row_nr", end_idx),
    ])

    # si per qualsevol motiu end queda abans que start, ho arreglem (swap)
    df_svaras = df_svaras.with_columns([
        pl.when(pl.col("_end_row_nr") < pl.col("_start_row_nr"))
          .then(pl.col("_start_row_nr"))
          .otherwise(pl.col("_end_row_nr"))
          .alias("_end_row_nr")
    ])

    # --- 2) marques puntuals ---
    df_start_marks = (
        df_svaras
        .with_columns(pl.col("_start_row_nr").alias("row_nr"))
        .filter((pl.col("row_nr") >= 0) & (pl.col("row_nr") < n))
        .select(["row_nr", pl.col("svara_label").alias("svara_start_label")])
    )

    df_end_marks = (
        df_svaras
        .with_columns(pl.col("_end_row_nr").alias("row_nr"))
        .filter((pl.col("row_nr") >= 0) & (pl.col("row_nr") < n))
        .select(["row_nr", pl.col("svara_label").alias("svara_end_label")])
    )

    # --- 3) svara_id expandit start..end (inclosos) ---
    df_svara_ranges = (
        df_svaras
        .with_row_index("svara_id")
        .with_columns(
            pl.int_ranges(
                pl.col("_start_row_nr").cast(pl.Int64),
                (pl.col("_end_row_nr") + 1).cast(pl.Int64),
            ).alias("row_nr")
        )
        .select(["row_nr", "svara_id"])
        .explode("row_nr")
    )

    df_svara_id = df_svara_ranges.select(["row_nr", "svara_id"])

    # --- 4) neteja i joins ---
    df_pitch = df_pitch.drop(
        [
            "svara_id", "svara_id_right",
            "svara_label", "svara_label_right",
            "svara_start_label", "svara_start_label_right",
            "svara_end_label", "svara_end_label_right",
        ],
        strict=False
    )

    df_pitch = (
        df_pitch
        .join(df_start_marks, on="row_nr", how="left")
        .join(df_end_marks, on="row_nr", how="left")
        .join(df_svara_id, on="row_nr", how="left")
        .drop("row_nr")
    )

    return df_pitch



'''
def attach_svara_annotations_to_pitch(
    df_pitch: pl.DataFrame,
    df_svaras: pl.DataFrame,
) -> pl.DataFrame:
    df_pitch = df_pitch.with_row_index("row_nr")

    t = df_pitch.get_column("time_rel_sec")
    n = df_pitch.height
    
    start_idx = t.search_sorted(df_svaras["start_time_sec"])
    end_idx = t.search_sorted(df_svaras["end_time_sec"], side="right") - 1
    
    df_svaras = df_svaras.with_columns([
        start_idx.alias("_start_idx"),
        end_idx.alias("_end_idx"),
    ])

    # --- Marcatge puntual svara_label (nullable) ---
    df_events = pl.concat([
        df_svaras
        .with_columns(pl.col("_after_idx").alias("row_nr"))
        .filter(pl.col("row_nr") < n)
        .select([
            "row_nr",
            (pl.col("svara_label") + pl.lit("_start")).alias("svara_label"),
        ]),
        df_svaras
        .with_columns(pl.col("_before_idx").alias("row_nr"))
        .filter(pl.col("row_nr") >= 0)
        .select([
            "row_nr",
            (pl.col("svara_label") + pl.lit("_end")).alias("svara_label"),
        ])
    ])

    df_marks = (
        df_events
        .group_by("row_nr")
        .agg(pl.col("svara_label").first())
    )

    # --- NOVETAT: svara_id expandit start..end (inclosos) ---
    df_svara_ranges = (
        df_svaras
        .with_row_index("svara_id")  # 0,1,2,... segons ordre
        .filter(
            (pl.col("_after_idx") < n) &
            (pl.col("_before_idx") >= 0) &
            (pl.col("_after_idx") <= pl.col("_before_idx"))
        )
        .with_columns(
            pl.int_ranges(
                pl.col("_after_idx").cast(pl.Int64),
                (pl.col("_before_idx") + 1).cast(pl.Int64)  # [start, end+1)
            ).alias("row_nr")
        )
        .select(["row_nr", "svara_id"])
        .explode("row_nr")
    )

    df_svara_id = (
        df_svara_ranges
        .group_by("row_nr")
        .agg(pl.col("svara_id").min())  # si mai hi ha solapaments, agafa el menor
    )

    # Neteja columnes prèvies conflictives (incloent *_right antics)
    df_pitch = df_pitch.drop(
        ["svara_label_right", "svara_id", "svara_id_right"],
        strict=False
    )

    # Join de marques + ids
    df_pitch = (
        df_pitch
        .join(df_marks, on="row_nr", how="left")
        .join(df_svara_id, on="row_nr", how="left")
        .drop("row_nr")
    )

    return df_pitch
'''