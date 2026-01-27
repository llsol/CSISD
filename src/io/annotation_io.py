import polars as pl
import pandas as pd
from pathlib import Path
from src.annotations.utils import time_str_to_sec

def load_annotations(
    file_path: Path | str,
    annotation_type: str,  # "svara" o "section"
    engine: str = "polars"
) -> pl.DataFrame:
    """
    Carrega anotacions (svara o secció) des d'un TSV.
    - Per *svara*: espera columnes "Begin time", "End time", "Annotation".
    - Per *secció*: espera format sense capçalera (start_time_sec, section_label).
    """
    if engine != "polars":
        raise NotImplementedError("Només Polars per simplificar.")

    if annotation_type == "svara":
        df = pl.read_csv(
            file_path,
            separator="\t",
            dtypes={"Begin time": pl.Utf8, "End time": pl.Utf8, "Annotation": pl.Utf8}
        ).drop("Tier")
        df = df.with_columns(
            pl.col("Begin time").map_elements(
                time_str_to_sec,
                return_dtype=pl.Float64  # <-- Especifica el tipus de retorn
            ).alias("start_time_sec"),
            pl.col("End time").map_elements(
                time_str_to_sec,
                return_dtype=pl.Float64  # <-- Especifica el tipus de retorn
            ).alias("end_time_sec"),
            pl.col("Annotation").alias("svara_label")
        )
    elif annotation_type == "section":
        df = pl.read_csv(
            file_path,
            separator="\t",
            has_header=False,
            new_columns=["start_time_sec", "section_label"],
            dtypes={"start_time_sec": pl.Float64, "section_label": pl.Utf8}
        )
    else:
        raise ValueError("annotation_type ha de ser 'svara' o 'section'.")

    return df





def save_annotations(
    df: pl.DataFrame,
    out_path: Path | str,
    format: str = "parquet",  # "parquet" o "tsv"
    **kwargs
) -> None:
    """
    Save annotations to Parquet or TSV.
    - If format="tsv", columns should be compatible with original format
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "parquet":
        df.write_parquet(out_path)
    elif format == "tsv":
        if "svara_label" in df.columns:
            df = df.rename({
                "start_time_sec": "Begin time",
                "end_time_sec": "End time",
                "svara_label": "Annotation"
            }).with_columns(
                pl.col("Begin time").cast(pl.Utf8).map_elements(lambda x: f"{x:.3f}"),
                pl.col("End time").cast(pl.Utf8).map_elements(lambda x: f"{x:.3f}")
            )
        df.write_csv(out_path, separator="\t")
    else:
        raise ValueError("Format has to be 'parquet' or 'tsv'.")
    

