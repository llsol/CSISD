
import polars as pl
import pandas as pd
from pathlib import Path
from src.annotations.utils import time_str_to_sec, validate_columns
from src.io.annotation_io import load_annotation_tsv

def load_svara_annotations(
        file_path: Path | str | None,
        engine='polars',
) -> pl.DataFrame | pd.DataFrame:
    """
    Load svara annotations from a .tsv file.

    Parameters
    ----------
    file_path : Path | str | None
        Path to the svara annotation .tsv file.
    engine : str
        'polars' or 'pandas' to specify the dataframe library.

    Returns
    -------
    DataFrame (Polars or Pandas)
    """

    if file_path is None:
        raise ValueError("file_path must be provided.")

    df_svaras = load_annotation_tsv(
        file_path=file_path,
        engine=engine,
        sep='\t',
    )

    required_cols = ["Begin time", "End time", "Annotation"]
    validate_columns(df_svaras, required_cols)

    col_dict = {
        'Begin time': 'start_time_sec',
        'End time': 'end_time_sec',
        'Duration': 'duration_sec',
        'Annotation': 'svara_label',
    }

    if engine == 'polars':
        df_svaras = df_svaras.drop("Tier", strict = False)
        df_svaras = df_svaras.with_columns(
            pl.col("Begin time").apply(time_str_to_sec).alias("Begin time"),
            pl.col("End time").apply(time_str_to_sec).alias("End time"),
        )
        df_svaras = df_svaras.rename(columns=col_dict, strict = False)

    elif engine == 'pandas':
        df_svaras = df_svaras.drop(columns=["Tier"], errors='ignore')
        df_svaras['Begin time'] = df_svaras['Begin time'].apply(time_str_to_sec)
        df_svaras['End time'] = df_svaras['End time'].apply(time_str_to_sec)
        df_svaras = df_svaras.rename(columns=col_dict, errors='ignore')

    return df_svaras




def attach_svara_annotations_to_pitch(df_pitch, df_svaras):

    """
    For each svara annotation, mark the closest start and end points in DataFrame.
    Adds/updates the 'svara_label' column.
    """
    df_pitch = df_pitch.copy()
    df_pitch["svara_label"] = ""

    for start, end, svara in df_svaras[["start_time_sec", "end_time_sec", "svara_label"]].values:
        
        # Closest time after or at start
        after_idx = df_pitch[df_pitch["time_rel_sec"] >= start]["time_rel_sec"].idxmin()
        df_pitch.at[after_idx, "svara_label"] += f"{svara}_start "
        
        # Closest time before end
        before_idx = df_pitch[df_pitch["time_rel_sec"] < end]["time_rel_sec"].idxmax()
        df_pitch.at[before_idx, "svara_label"] += f"{svara}_end "

    # Remove trailing spaces
    df_pitch["svara_label"] = df_pitch["svara_label"].str.strip()

    return df_pitch