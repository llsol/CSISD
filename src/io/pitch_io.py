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

    if isinstance(file_path, str):
        file_path = Path(file_path)

    ext = file_path.suffix.lower()


    if engine == 'polars':

        if ext == '.parquet':
            df = pl.read_parquet(file_path)

        elif ext == '.tsv':
            if column_names is None:
                df = pl.read_csv(file_path, separator=sep, has_header=True)
            else:
                df = pl.read_csv(file_path, separator=sep, has_header=False)
                df = df.rename({old: new for old, new in zip(df.columns, column_names)})
        
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
    """
    Generic saver for pitch files in .tsv or .parquet format.
    """

    if isinstance(file_path, str):
        file_path = Path(file_path)

    ext = file_path.suffix.lower()


    if engine == 'polars':

        if ext == '.parquet':
            df.write_parquet(file_path)

        elif ext == '.tsv':
            df.write_csv(file_path)
        
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