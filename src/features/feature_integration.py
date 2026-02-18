import numpy as np
import polars as pl
from src.features.derivatives import compute_derivatives
from src.features.basic_features import compute_base_features
from src.features.derivatives import compute_derivative_features
from src.annotations.utils import time_str_to_sec

def compute_svara_segment_features(
    df_pitch: pl.DataFrame,
    df_svaras: pl.DataFrame,
    recording_id: str,
    col_time: str = "time_rel_sec",
    col_pitch: str = "f0_savgol_p3_w13",
    col_deriv_in: str = "f0_savgol_p3_w13",
    col_begin: str = "start_time_sec",
    col_end: str = "end_time_sec",
    col_label: str = "svara_label",
    deriv_window_length: int = 13,
    deriv_polyorder: int = 3,
) -> pl.DataFrame:
    """
    1 fila per segment (svara) amb:
      - base features sobre col_pitch
      - derivative features sobre d1/d2 (derivades de col_deriv_in)
    """
    # 1) calcula d1/d2 global i afegeix-los al df_pitch
    der = compute_derivatives(
        df_pitch,
        col_in=col_deriv_in,
        window_length=deriv_window_length,
        polyorder=deriv_polyorder,
    )

    df_pitch = df_pitch.with_columns([
        pl.Series("d1", der["d1"]),
        pl.Series("d2", der["d2"]),
    ])

    t_all = df_pitch[col_time].to_numpy()
    rows = []

    for segment_id, row in enumerate(df_svaras.iter_rows(named=True)):
        t_start = time_str_to_sec(row[col_begin])
        t_end = time_str_to_sec(row[col_end])
        if t_end < t_start:
            t_start, t_end = t_end, t_start

        svara_label = row[col_label]

        mask_seg = (t_all >= t_start) & (t_all <= t_end)
        df_seg = df_pitch.filter(pl.Series(mask_seg))

        out = {
            "recording_id": recording_id,
            "segment_id": segment_id,
            "svara_label": svara_label,
            "t_start": float(t_start),
            "t_end": float(t_end),
            "duration_sec": float(t_end - t_start),
            "n_frames": int(mask_seg.sum()),
        }

        out.update(compute_base_features(df_seg, pitch_column=col_pitch, time_column=col_time))
        out.update(compute_derivative_features(df_seg, col_d1="d1", col_d2="d2"))

        rows.append(out)

    return pl.DataFrame(rows)
