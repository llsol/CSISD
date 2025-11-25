import polars as pl
import pandas as pd
from pathlib import Path

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

    ext = file_path.suffix.lower()

    mapping = dict(zip(df.columns, column_names))

    if engine == 'polars':

        if ext == '.parquet':
            df = pl.read_parquet(file_path)

        elif ext == '.tsv':
            if column_names is None:
                df = pl.read_csv(file_path, separator=sep, has_header=True)
            else:
                df = pl.read_csv(file_path, separator=sep, has_header=False)
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
        root_dir: Path | str = "data/interim"
) -> pl.DataFrame:
    """
    Load a preprocessed pitch parquet for a given recording_id
    from data/interim/<recording_id>/pitch/.
    """

    if isinstance(root_dir, str):
        root_dir = Path(root_dir)

    file_path = root_dir / recording_id / "pitch" / f"{recording_id}_pitch_preprocessed.parquet"

    if not file_path.exists():
        raise FileNotFoundError(f"No preprocessed pitch file found at {file_path}")

    return pl.read_parquet(file_path)





def save_preprocessed_pitch(
        df: pl.DataFrame,
        recording_id: str,
        root_dir: Path | str = "data/interim"
) -> Path:
    """
    Save the preprocessed pitch dataframe as a Parquet file
    inside data/interim/<recording_id>/pitch/.

    Drops internal/temporary columns before saving:
        - group_id
        - valid_for_spline
    """

    if isinstance(root_dir, str):
        root_dir = Path(root_dir)

    # Output directory
    out_dir = root_dir / recording_id / "pitch"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Output file
    out_path = out_dir / f"{recording_id}_pitch_preprocessed.parquet"

    # Remove internal columns if present
    drop_cols = [c for c in ["group_id", "valid_for_spline"] if c in df.columns]
    df_to_save = df.drop(drop_cols)

    # Save parquet
    df_to_save.write_parquet(out_path)

    return out_path