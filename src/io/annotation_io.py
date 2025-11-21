import polars as pl
import pandas as pd
from pathlib import Path


def load_annotation_tsv(
        file_path: Path | str,
        engine: str = "polars",
        sep: str = "\t",
):
    """
    Load an annotation .tsv file (svara, section, or any annotation)
    and return a dataframe in either Polars or Pandas format.

    Returns
    -------
    DataFrame (Polars or Pandas)
    """

    if isinstance(file_path, str):
        file_path = Path(file_path)

    ext = file_path.suffix.lower()
    if ext != ".tsv":
        raise ValueError(
            f"load_annotation_tsv only supports .tsv files. Got: {ext}"
        )

    if engine == "polars":
        df = pl.read_csv(file_path, separator=sep)
        return df

    elif engine == "pandas":
        df = pd.read_csv(file_path, sep=sep)
        return df

    else:
        raise ValueError("engine must be either 'polars' or 'pandas'.")


