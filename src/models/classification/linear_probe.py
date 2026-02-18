from pathlib import Path
import numpy as np
import polars as pl


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.io.pitch_io import load_preprocessed_pitch
from src.io.annotation_io import load_annotations
from src.features.feature_integration import compute_svara_segment_features
import settings as S

RECORDINGS = S.SARASUDA_VARNAM




# ---- importa els teus mètodes (ajusta imports segons el teu projecte) ----
# from features_base import compute_base_features
# from features_derivatives import compute_derivatives, compute_derivative_features
# from segment_features import compute_svara_segment_features
# from utils import time_str_to_sec

# Aquí assumeixo que compute_svara_segment_features ja existeix (el que vam definir).
# i que time_str_to_sec també el tens.





def df_features_to_xy(
    df_feat: pl.DataFrame,
    y_col: str = "svara_label",
    drop_cols: tuple[str, ...] = ("recording_id", "segment_id", "svara_label"),
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Converteix el df de features (1 fila per segment) a X, y.
    """
    # neteja files sense etiqueta
    df_feat = df_feat.filter(pl.col(y_col).is_not_null())

    # quines columnes són features?
    feature_cols = [c for c in df_feat.columns if c not in drop_cols]
    X = df_feat.select(feature_cols).to_numpy()
    y = df_feat.get_column(y_col).to_numpy()

    return X, y, feature_cols


def build_dataset(
    corpus_root: Path,
    recording_id_list: list[str],
) -> tuple[np.ndarray, np.ndarray, pl.DataFrame, list[str]]:
    """
    Itera recordings, computa features per segment i concatena:
      X_all: (n_segments_total, n_features)
      y_all: (n_segments_total,)
      df_meta_all: (n_segments_total, ...) inclou recording_id/segment_id etc.
      feature_cols: llista de features (mateixa per tots)
    """
    X_list = []
    y_list = []
    meta_list = []
    feature_cols_ref = None

    for recording_id in recording_id_list:
        df_pitch = load_preprocessed_pitch(corpus_root, recording_id)
        df_svaras = load_annotations(corpus_root, recording_id)

        # ---- IMPORTANT ----
        # Aquí assumeixo que df_pitch JA té group_id i f0_pchip (per derivades),
        # i f0_savgol_p3_w13 (per base features).
        # Si no, abans d’això hauràs de passar preprocess_pitch(df_pitch_original).

        df_feat = compute_svara_segment_features(
            df_pitch=df_pitch,
            df_svaras=df_svaras,
            recording_id=recording_id,
            col_time="time_rel_sec",
            col_pitch="f0_savgol_p3_w13",
            col_deriv_in="f0_savgol_p3_w13",
            col_begin="start_time_sec",
            col_end="end_time_sec",
            col_label="svara_label",
            deriv_window_length=13,
            deriv_polyorder=3,
        )

        X, y, feature_cols = df_features_to_xy(df_feat, y_col="svara_label")

        if feature_cols_ref is None:
            feature_cols_ref = feature_cols
        elif feature_cols != feature_cols_ref:
            raise ValueError("Les columnes de features no coincideixen entre recordings.")

        X_list.append(X)
        y_list.append(y)

        meta_list.append(df_feat.select(["recording_id", "segment_id", "svara_label", "t_start", "t_end", "duration_sec"]))

    X_all = np.vstack(X_list) if X_list else np.empty((0, 0), dtype=float)
    y_all = np.concatenate(y_list) if y_list else np.empty((0,), dtype=object)
    df_meta_all = pl.concat(meta_list) if meta_list else pl.DataFrame()

    return X_all, y_all, df_meta_all, (feature_cols_ref or [])


def run_linear_probe(X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1_macro")
    return float(scores.mean()), float(scores.std())


def main():
    corpus_root = Path("data/corpus")

    # posa aquí els recording_ids que vulguis
    recording_id_list = RECORDINGS

    X, y, df_meta, feature_cols = build_dataset(corpus_root, recording_id_list)

    print("X shape:", X.shape)
    print("n classes:", len(np.unique(y)))

    mean_f1, std_f1 = run_linear_probe(X, y)
    print("F1_macro mean:", mean_f1)
    print("F1_macro std:", std_f1)


if __name__ == "__main__":
    main()
