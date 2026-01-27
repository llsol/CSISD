import polars as pl
import pandas as pd
from pathlib import Path
import numpy as np
from settings import PROJECT_ROOT




def _resolve_path(p: Path | str) -> Path:
    p = Path(p)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p





def load_pitch_file(
        file_path: Path | str,
        engine='polars',
        sep='\t',
        column_names=None,
):
    """
    Generic loader for pitch files in .tsv or .parquet format.

    Behaviour:
    - If column_names=None, it assumes file has header.
    - If column_names is provided, it treats file as headerless and assigns these names.
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)
    
    file_path = _resolve_path(file_path)

    ext = file_path.suffix.lower()


    if engine == 'polars':

        if ext == '.parquet':
            df = pl.read_parquet(file_path)

        elif ext == '.tsv':
            if column_names is None:
                df = pl.read_csv(file_path, separator=sep, has_header=True)
            else:
                df = pl.read_csv(file_path, separator=sep, has_header=False)
                mapping = dict(zip(df.columns, column_names))
                df = df.rename(mapping)

        else:
            raise ValueError(f"Unsupported file extension: {ext}")


    elif engine == 'pandas':

        if ext == '.parquet':
            df = pd.read_parquet(file_path)

        elif ext == '.tsv':
            if column_names is None:
                df = pd.read_csv(file_path, sep=sep, header=0)
            else:
                df = pd.read_csv(file_path, sep=sep, header=None, names=column_names)
        
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    else:
        raise ValueError("Engine must be 'polars' or 'pandas'.")

    return df





def save_pitch_file(
        df,
        file_path: Path | str,
        engine='polars',
        sep='\t',
):

    file_path = _resolve_path(file_path)

    ext = file_path.suffix.lower()

    if engine == 'polars':

        if ext == '.parquet':
            df.write_parquet(file_path)

        elif ext == '.tsv':
            df.write_csv(file_path, separator=sep)
        
        else:
            raise ValueError(f"Unsupported file extension: {ext}")


    elif engine == 'pandas':

        if ext == '.parquet':
            df.to_parquet(file_path)

        elif ext == '.tsv':
            df.to_csv(file_path, sep=sep, index=False)

        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    else:
        raise ValueError("Engine must be 'polars' or 'pandas'.")

    return file_path





def load_preprocessed_pitch(
        recording_id: str,
        root_dir: Path | str = "data/interim",
        tonic_hz: float = None,
        convert_to_cents: bool = True
):
    """
    Load a preprocessed pitch parquet for a given recording_id
    from data/interim/<recording_id>/pitch/.
    """

    if isinstance(root_dir, str):
        root_dir = Path(root_dir)

    root_dir = _resolve_path(root_dir)

    file_path = (
        root_dir
        / recording_id
        / "pitch"
        / f"{recording_id}_pitch_preprocessed.parquet"
    )

    if not file_path.exists():
        raise FileNotFoundError(f"No preprocessed pitch file found at {file_path}")

    df = pl.read_parquet(file_path)

    if convert_to_cents and tonic_hz is None:
        raise ValueError("convert_to_cents=True requires tonic_hz")

    if convert_to_cents:
        pitch_cols = [
            "f0_Hz",
            "f0_interpolated",
            "f0_pchip",
            "f0_savgol_p3_w13",
            "f0_savgol_p3_w27"
        ]

        for col in pitch_cols:
            if col in df.columns:
                f_hz = df[col].to_numpy()
                mask = np.isfinite(f_hz) & (f_hz > 0)
                f_cents = np.full_like(f_hz, np.nan)
                f_cents[mask] = 1200.0 * np.log2(f_hz[mask] / tonic_hz)
                df = df.with_columns(
                    pl.Series(f"{col}_cents", f_cents)
                )

    return df



#def load_preprocessed_pitch(
#        recording_id: str,
#        root_dir: Path | str = "data/interim",
#        tonic_hz: float = None,
#        convert_to_cents: bool = True
#) -> pl.DataFrame:
#    """
#    Load a preprocessed pitch parquet for a given recording_id
#    from data/interim/<recording_id>/pitch/.
#    """
#
#    if isinstance(root_dir, str):
#        root_dir = Path(root_dir)
#
#    root_dir = _resolve_path(root_dir)
#
#    file_path = root_dir / recording_id / "pitch" / f"{recording_id}_pitch_preprocessed.parquet"
#
#    if not file_path.exists():
#        raise FileNotFoundError(f"No preprocessed pitch file found at {file_path}")
#
#    df =pl.read_parquet(file_path)
#
#    candidates_clean_pitch = [
#        "f0_savgol_p3_w13",
#        "f0_pchip",
#        "f0_Hz",
#    ]
#
#    clean_pitch = None
#
#    for col in candidates_clean_pitch:
#        if col in df.columns:
#            clean_pitch = col
#            break
#    
#    
#    if convert_to_cents:
#
#        pitch_cols = [
#            "f0_Hz",
#            "f0_interpolated",
#            "f0_pchip",
#            "f0_savgol_p3_w13",
#            "f0_savgol_p3_w27",
#        ]
#
#        for col in pitch_cols:
#            if col in df.columns:
#                f_hz = df[col].to_numpy()
#                mask = np.isfinite(f_hz) & (f_hz > 0)
#                f_cents = np.full_like(f_hz, np.nan)
#                f_cents[mask] = 1200.0 * np.log2(f_hz[mask] / tonic_hz)
#                df = df.with_columns(
#                    pl.Series(f"{col}_cents", f_cents)
#                )
#
#    else:
#        clean_pitch = None
#
#        for col in candidates_clean_pitch:
#            if col in df.columns:
#                clean_pitch = col
#                break
#        
#        if clean_pitch is None:
#            raise ValueError("No suitable clean pitch column found in the dataframe.")
#
#        else:
#            f_hz = df[clean_pitch].to_numpy()
#            mask = np.isfinite(f_hz) & (f_hz > 0)
#            f_cents = np.full_like(f_hz, np.nan)
#            f_cents[mask] = 1200 * np.log2(f_hz[mask] / tonic_hz)
#            df = df.with_columns(
#                pl.Series(f"{clean_pitch}_cents", f_cents)
#            )
#            
#
#        
#    return df



def save_preprocessed_pitch(
        df: pl.DataFrame,
        recording_id: str,
        root_dir: Path | str = "data/interim",
        debug: bool = False,
        create_tsv: bool = False
) -> Path:
    """
    Save the preprocessed pitch dataframe as a Parquet file
    inside data/interim/<recording_id>/pitch/.

    Only selected columns are saved based on whether debug is True or False.
    """

    if isinstance(root_dir, str):
        root_dir = Path(root_dir)

    root_dir = _resolve_path(root_dir)

    # Output directory
    out_dir = root_dir / recording_id / "pitch"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Output file
    out_path = out_dir / f"{recording_id}_pitch_preprocessed.parquet"

    production_cols = [
        "time_rel_sec",
        "f0_Hz",
        "group_id",
        "svara_id",
        "svara_start_label",
        "svara_end_label",
    ]
    if "f0_pchip" in df.columns:
        production_cols.append("f0_pchip")

    if "f0_savgol_p3_w13" in df.columns:
        production_cols.append("f0_savgol_p3_w13")

    if "f0_savgol_p3_w27" in df.columns:
        production_cols.append("f0_savgol_p3_w27")

    debug_cols = [
        "f0_interpolated",
        "too_long_to_interp",
        "is_outlier",
        "valid_for_pchip",
    ]

    if debug:
        out_path = out_dir / f"{recording_id}_pitch_preprocessed_debug.parquet"
        for c in debug_cols:
            if c in df.columns:
                production_cols.append(c)

    final_cols = [c for c in production_cols if c in df.columns]
    df_out = df.select(final_cols)

    df_out = df_out.with_columns([
        pl.col(pl.Float64).cast(pl.Float32)
    ])
    # Save parquet
    df_out.write_parquet(out_path)

    if create_tsv:
        tsv_path = str(out_path).replace(".parquet", ".tsv")
        df_out.write_csv(tsv_path, separator="\t")
    
    return out_path




def save_flat_regions(
        df: pl.DataFrame,
        recording_id: str,
        root_dir: Path | str = "data/interim",
):
    """
    Save pitch + flat_region for plotting / inspection.
    """

    if isinstance(root_dir, str):
        root_dir = Path(root_dir)

    root_dir = _resolve_path(root_dir)

    out_dir = root_dir / recording_id / "flat_regions"
    out_dir.mkdir(parents=True, exist_ok=True)

    file_path = out_dir / f"{recording_id}_flat_regions.parquet"


    # COLUMNS TO SAVE

    SAVE_COLS = [
        "time_rel_sec",

        # pitch (tria les que t'interessin per plots)
        "f0_savgol_p3_w13",
        "f0_savgol_p3_w13_cents",

        # flatness
        "flat_region",

        "svara_id"
    ]

    cols_present = [c for c in SAVE_COLS if c in df.columns]
    df_out = df.select(cols_present)

    df_out.write_parquet(file_path)
    return file_path