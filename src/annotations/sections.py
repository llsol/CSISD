import polars as pl
import numpy as np

def attach_section_annotations_to_pitch(
    df_pitch: pl.DataFrame,
    df_sections: pl.DataFrame,
    threshold_sec: float = 1.0
) -> pl.DataFrame:
    """
    Assigna etiquetes de secció al DataFrame de pitch.
    Usa les marques de svara si estan disponibles i són properes.
    """
    df_pitch = df_pitch.clone().with_row_index("row_nr")

    pitch_times = df_pitch["time_rel_sec"].to_numpy()

    # Extreu temps de svara_start si existeixen
    svara_start_times = (
        df_pitch
        .filter(pl.col("svara_label").fill_null("").str.ends_with("_start"))
        ["time_rel_sec"]
        .to_numpy()
    )

    # Inicialitza columna (nullable string)
    df_pitch = df_pitch.with_columns(pl.lit(None).cast(pl.Utf8).alias("section_label"))

    for t_sec, label in df_sections[["start_time_sec", "section_label"]].rows():
        if svara_start_times.size > 0:
            nearest_idx = np.abs(svara_start_times - t_sec).argmin()
            if abs(svara_start_times[nearest_idx] - t_sec) <= threshold_sec:
                idx_pitch = np.abs(pitch_times - svara_start_times[nearest_idx]).argmin()
            else:
                idx_pitch = np.abs(pitch_times - t_sec).argmin()
        else:
            idx_pitch = np.abs(pitch_times - t_sec).argmin()

        # Assigna etiqueta per índex (no per float equality)
        df_pitch = df_pitch.with_columns(
            pl.when(pl.col("row_nr") == int(idx_pitch))
              .then(pl.lit(label))
              .otherwise(pl.col("section_label"))
              .alias("section_label")
        )

    return df_pitch.drop("row_nr")

'''
def attach_section_annotations_to_pitch(
    df_pitch: pl.DataFrame,
    df_sections: pl.DataFrame,
    threshold_sec: float = 1.0
) -> pl.DataFrame:
    """
    Assigna etiquetes de secció al DataFrame de pitch.
    Usa les marques de svara si estan disponibles i són properes.
    """
    df_pitch = df_pitch.clone()
    pitch_times = df_pitch["time_rel_sec"].to_numpy()

    # Extreu temps de svara_start si existeixen
    svara_start_times = (
        df_pitch.filter(pl.col("svara_label").str.ends_with("_start"))["time_rel_sec"]
        .to_numpy()
    )

    # Inicialitza columna
    df_pitch = df_pitch.with_columns(pl.lit(None).alias("section_label"))

    for t_sec, label in df_sections[["start_time_sec", "section_label"]].rows():
        if svara_start_times.size > 0:
            # Busca el svara_start més proper
            nearest_idx = np.abs(svara_start_times - t_sec).argmin()
            if abs(svara_start_times[nearest_idx] - t_sec) <= threshold_sec:
                idx_pitch = np.abs(pitch_times - svara_start_times[nearest_idx]).argmin()
            else:
                idx_pitch = np.abs(pitch_times - t_sec).argmin()
        else:
            idx_pitch = np.abs(pitch_times - t_sec).argmin()

        # Assigna etiqueta
        df_pitch = df_pitch.with_columns(
            pl.when(pl.col("time_rel_sec") == pitch_times[idx_pitch])
              .then(pl.lit(label))
              .otherwise(pl.col("section_label"))
              .alias("section_label")
        )

    return df_pitch


'''
"""
import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from src.io.annotation_io import load_annotation_tsv




def load_section_annotations(
        file_path: Path | str | None,
        engine='polars',
) -> pl.DataFrame | pd.DataFrame:
    '''
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
    '''

    if file_path is None:
        raise ValueError("file_path must be provided.")

    df_sections = load_annotation_tsv(
        file_path=file_path,
        engine=engine,
        sep='\t',
        section_annotation=True
    )

    return df_sections



#def attach_section_annotations_to_pitch(df_pitch, df_sections, threshold_sec=1.0):
    '''
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
    '''

def attach_section_annotations_to_pitch(df_pitch: pl.DataFrame, 
                                        df_sections: pl.DataFrame, 
                                        threshold_sec: float, 
                                        verbose = False
):
    # Work on a copy for safety
    df_pitch = df_pitch.clone()

    # Ensure the column exists
    if "section_label" not in df_pitch.columns:
        df_pitch = df_pitch.with_columns(
            pl.lit(None).alias("section_label")
        )

    # Vectorized arrays for speed
    pitch_times = df_pitch["time_rel_sec"].to_numpy()

    # Extract svara start times if present
    if "svara_mark" in df_pitch.columns:
        mask_starts = df_pitch.filter(
            pl.col("svara_mark").cast(pl.Utf8).str.ends_with("_start")
        )["time_rel_sec"]
        svara_start_times = mask_starts.to_numpy()
    else:
        svara_start_times = np.array([])

    counter = 1

    # Iterate through each section annotation
    for row in df_sections.select(["start_time_sec", "section_label"]).rows():
        t_sec, label = row

        # No svara information. Direct nearest pitch alignment
        if svara_start_times.size == 0:
            idx_pitch = np.abs(pitch_times - t_sec).argmin()
            df_pitch = df_pitch.with_columns(
                pl.when(pl.col("time_rel_sec") == pitch_times[idx_pitch])
                  .then(pl.lit(label))
                  .otherwise(pl.col("section_label"))
                  .alias("section_label")
            )
            if verbose:
                print(f'({counter})Svara mark closer than {threshold_sec} second(s) for section {label} at time mark {t_sec} not found!')
            continue

        # Compute nearest svara start
        idx_svara = np.abs(svara_start_times - t_sec).argmin()
        nearest_svara_time = svara_start_times[idx_svara]

        # If svara is close enough → use svara start
        if abs(nearest_svara_time - t_sec) <= threshold_sec:
            idx_pitch = np.abs(pitch_times - nearest_svara_time).argmin()
            df_pitch = df_pitch.with_columns(
                pl.when(pl.col("time_rel_sec") == pitch_times[idx_pitch])
                  .then(pl.lit(label))
                  .otherwise(pl.col("section_label"))
                  .alias("section_label")
            )
            if verbose:
                print(f'({counter}) Section {label} at time mark {t_sec} assigned to time mark {nearest_svara_time}, corresponding to a svara start. ')
            
        else:
            # Fallback → use nearest pitch timestamp
            idx_pitch = np.abs(pitch_times - t_sec).argmin()
            df_pitch = df_pitch.with_columns(
                pl.when(pl.col("time_rel_sec") == pitch_times[idx_pitch])
                  .then(pl.lit(label))
                  .otherwise(pl.col("section_label"))
                  .alias("section_label")
            )
            if verbose:
                print(f'({counter}) Section {label} at time mark {t_sec} assigned to time mark {t_sec}. ')
            
        counter += 1

    return df_pitch


def save_section_annotations_parquet(
    file_path: Path | str,
    out_path: Path | str,
    engine: str = "polars",
    sep: str = "\t",
) -> pl.DataFrame | pd.DataFrame:
    '''
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
    '''
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
"""