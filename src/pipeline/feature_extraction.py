import numpy as np
import polars as pl
from settings import SARASUDA_VARNAM
from pathlib import Path

from src.io.pitch_io import load_pitch_file
from src.features.basic_features import compute_base_features
from src.features.derivatives import compute_derivatives, compute_derivative_features


global_feature_functions = [
    compute_derivatives
]

svara_feature_functions = [
    compute_base_features,
    compute_derivative_features,
]

def feature_extraction_one_recording(
    recording_id: str,
    pitch_column: str = "f0_savgol_p3_w13",
    out_path: Path = None,
) -> pl.DataFrame:

    preprocessed_path = Path(f"data/interim/{recording_id}/pitch/{recording_id}_pitch_preprocessed.parquet")
    df_prep = load_pitch_file(preprocessed_path)

    # 1) afegeix globals (IMPORTANT: assigna el retorn)
    df_prep = add_global_features_to_pitch(
        df_pitch=df_prep,
        global_feature_functions=global_feature_functions,
    )

    # 2) features per svara_id (IMPORTANT: passa recording_id i feature_functions)
    df_features = compute_features_by_svara_id(
        df_pitch=df_prep,
        recording_id=recording_id,
        pitch_column=pitch_column,
        feature_functions=svara_feature_functions,
    )

    # 3) desa (1 recording -> 1 parquet)
    if out_path is None:
        out_path = Path(f"data/interim/{recording_id}/features/{recording_id}_svara_features.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.write_parquet(out_path)

    print(f" Features guardades: {out_path}")
    return df_features


def feature_extraction_all_recordings(recording_ids: list = SARASUDA_VARNAM):
    for recording_id in recording_ids:
        print(f" Feature extraction {recording_id}...")
        feature_extraction_one_recording(recording_id=recording_id)
    print("\n All recordings feature-extracted.")


def add_global_features_to_pitch(
    df_pitch: pl.DataFrame,
    global_feature_functions: list,
) -> pl.DataFrame:
    """
    Aplica funcions globals (sobre tota la pitchcurve) i les afegeix com columnes a df_pitch.

    Cada funció ha de retornar un dict:
      - valor array/list (len == df_pitch.height): s'afegeix com columna per fila
      - valor escalar: s'afegeix com columna constant (broadcast)
    """
    df_pitch = df_pitch.clone()
    n = df_pitch.height

    new_cols = []

    for fn in global_feature_functions:
        out = fn(df_pitch)
        if not isinstance(out, dict):
            raise ValueError(f"{fn.__name__} ha de retornar un dict")

        for k, v in out.items():
            if isinstance(v, (np.ndarray, list, tuple)):
                if len(v) != n:
                    raise ValueError(f"Feature '{k}' té len={len(v)} però df_pitch.height={n}")
                v_arr = np.asarray(v)
                new_cols.append(pl.Series(name=k, values=v_arr))
            else:
                if v is None:
                    new_cols.append(pl.lit(None).alias(k))
                else:
                    new_cols.append(pl.lit(v).alias(k))

    if new_cols:
        df_pitch = df_pitch.with_columns(new_cols)

    return df_pitch


def compute_features_by_svara_id(
    df_pitch: pl.DataFrame,
    recording_id: str,
    pitch_column: str,
    feature_functions: list,
) -> pl.DataFrame:
    """
    Computa features per cada svara_id (una fila per segment).
    Les feature_functions han de retornar dicts.
    """
    if "svara_id" not in df_pitch.columns:
        raise ValueError("df_pitch no té columna 'svara_id'")
    if "time_rel_sec" not in df_pitch.columns:
        raise ValueError("df_pitch no té columna 'time_rel_sec'")
    if pitch_column not in df_pitch.columns:
        raise ValueError(f"df_pitch no té columna '{pitch_column}'")

    rows = []

    svara_ids = (
        df_pitch
        .select(pl.col("svara_id").drop_nulls().unique().sort())
        .get_column("svara_id")
        .to_list()
    )

    for svara_id in svara_ids:
        df_svara = df_pitch.filter(pl.col("svara_id") == svara_id)
        if df_svara.is_empty():
            continue

        svara_start = df_svara["time_rel_sec"].min()
        svara_end = df_svara["time_rel_sec"].max()
        svara_duration = svara_end - svara_start
        svara_n = df_svara.height

        features = {}
        for fn in feature_functions:
            # compute_base_features ara accepta df directament
            out = fn(df_svara)
            if not isinstance(out, dict):
                raise ValueError(f"{fn.__name__} ha de retornar un dict")
            features.update(out)

        row = {
            "recording_id": recording_id,
            "svara_id": svara_id,
            "svara_start_sec": float(svara_start) if svara_start is not None else np.nan,
            "svara_end_sec": float(svara_end) if svara_end is not None else np.nan,
            "svara_duration_sec": float(svara_duration) if svara_duration is not None else np.nan,
            "svara_n_rows": int(svara_n),
            **features,
        }

        rows.append(row)

    return pl.DataFrame(rows)


if __name__ == "__main__":
    feature_extraction_one_recording(recording_id=SARASUDA_VARNAM[0])
