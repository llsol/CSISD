import polars as pl
import pandas as pd
from pathlib import Path


def load_annotation_tsv(
        file_path: Path | str,
        engine: str = "polars",
        sep: str = "\t",
        section_annotation = False
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
        if not section_annotation:
            df = pl.read_csv(file_path, separator=sep)
        
        else:
            df = pl.read_csv(
                file_path, 
                separator=sep, 
                has_header=False, 
                new_columns=["start_time_sec","section_label"], 
                infer_schema_length=0
            ) 

            df = df.with_columns(pl.col("start_time_sec").cast(pl.Float64)) 

        return df

    elif engine == "pandas":
        if not section_annotation:
            df = pd.read_csv(file_path, sep=sep)
        
        else: 
            df = pd.read_csv(
                file_path,
                sep=sep,
                header=None,  
                names=["start_time_sec", "section_label"]
            )
            
            df["start_time_sec"] = df["start_time_sec"].astype(float)
                    
        return df

    else:
        raise ValueError("engine must be either 'polars' or 'pandas'.")


