
import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from src.io.annotation_io import load_annotation_tsv




def load_section_annotations(
        file_path: Path | str | None,
        engine='polars',
) -> pl.DataFrame | pd.DataFrame:
    """
    Load section annotations from a .tsv file.

    Parameters
    ----------
    file_path : Path | str | None
        Path to the section annotation .tsv file.
    engine : str
        'polars' or 'pandas' to specify the dataframe library.

    Returns
    -------
    DataFrame (Polars or Pandas)
    """

    if file_path is None:
        raise ValueError("file_path must be provided.")

    df_sections = load_annotation_tsv(
        file_path=file_path,
        engine=engine,
        sep='\t',
    )


    if engine == 'polars':
        df_sections = df_sections.rename({old: new for old, new in zip(df_sections.columns, ["start_time_sec", "section_label"])})
        df_sections = df_sections.with_columns(pl.col("start_time_sec").cast(pl.Float64))

    elif engine == 'pandas':
        df_sections.columns = ["start_time_sec", "section_label"]
        df_sections["start_time_sec"] = df_sections["start_time_sec"].astype(float)

    return df_sections



def attach_section_annotations_to_pitch(df_pitch, df_sections, threshold_sec=1.0):
    """
    Attach section annotations to pitch DataFrame.
    Uses svara starts if available and close enough;
    otherwise uses the closest general pitch timestamp.

    Parameters
    ----------
    df_pitch : pandas.DataFrame
        Must contain 'time_rel_sec' and (optionally) 'svara_label'.
    df_sections : pandas.DataFrame
        Must contain 'start_time_sec' and 'section_label'.
    threshold_sec : float
        Maximum distance (in seconds) from a svara start for alignment.

    Returns
    -------
    pandas.DataFrame
        df_pitch with a new/updated 'section_label' column.
    """

    # Work on a copy for safety
    df_pitch = df_pitch.copy()

    # Ensure the column exists
    if "section_label" not in df_pitch.columns:
        df_pitch["section_label"] = ""

    # Vectorized arrays for speed
    pitch_times = df_pitch["time_rel_sec"].to_numpy()

    # Extract svara start times if present
    if "svara_label" in df_pitch.columns:
        mask_starts = df_pitch["svara_label"].astype(str).str.endswith("_start")
        svara_start_times = df_pitch.loc[mask_starts, "time_rel_sec"].to_numpy()
    else:
        svara_start_times = np.array([])

    # Iterate through each section annotation
    for t_sec, label in df_sections[["start_time_sec", "section_label"]].values:

        # No svara information. Direct nearest pitch alignment
        if svara_start_times.size == 0:
            idx_pitch = np.abs(pitch_times - t_sec).argmin()
            df_pitch.at[idx_pitch, "section_label"] = label
            continue

        # Compute nearest svara start
        idx_svara = np.abs(svara_start_times - t_sec).argmin()
        nearest_svara = svara_start_times[idx_svara]

        # If svara is close enough → use svara start
        if abs(nearest_svara - t_sec) <= threshold_sec:
            idx_pitch = np.abs(pitch_times - nearest_svara).argmin()
            df_pitch.at[idx_pitch, "section_label"] = label

        else:
            # Fallback → use nearest pitch timestamp
            idx_pitch = np.abs(pitch_times - t_sec).argmin()
            df_pitch.at[idx_pitch, "section_label"] = label

    return df_pitch


def save_section_annotations_parquet(
    file_path: Path | str,
    out_path: Path | str,
    engine: str = "polars",
    sep: str = "\t",
) -> pl.DataFrame | pd.DataFrame:
    """
    Load section annotations from a headerless TSV (time_sec, label) and save as parquet.

    Parameters
    ----------
    file_path : Path | str
        Path to headerless section TSV (two columns: start_time_sec, section_label).
    out_path : Path | str
        Path to save parquet.
    engine : str
        'polars' or 'pandas'.
    sep : str
        TSV separator (default '\\t').

    Returns
    -------
    DataFrame (Polars or Pandas) saved to parquet.
    """
    file_path = Path(file_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if engine == "polars":
        # headerless read
        df_sections = pl.read_csv(
            file_path,
            separator=sep,
            has_header=False,
            new_columns=["start_time_sec", "section_label"],
        ).with_columns(
            pl.col("start_time_sec").cast(pl.Float64),
            pl.col("section_label").cast(pl.Utf8),
        ).sort("start_time_sec")

        df_sections.write_parquet(out_path)
        return df_sections

    elif engine == "pandas":
        df_sections = pd.read_csv(
            file_path,
            sep=sep,
            header=None,
            names=["start_time_sec", "section_label"],
        )
        df_sections["start_time_sec"] = df_sections["start_time_sec"].astype(float)
        df_sections["section_label"] = df_sections["section_label"].astype(str)

        df_sections = df_sections.sort_values("start_time_sec").reset_index(drop=True)
        df_sections.to_parquet(out_path, index=False)
        return df_sections

    else:
        raise ValueError("engine must be 'polars' or 'pandas'.")