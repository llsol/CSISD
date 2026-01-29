
import numpy as np
import polars as pl
from scipy.signal import find_peaks

import settings as S


def extract_peaks(
    df: pl.DataFrame,
    *,
    recording_id: str,
    tonic_hz: float,
    time_col: str = S.TIME_COL,
    savgol_cents_col: str = S.PITCH_COL_CENTS,
    raw_hz_col: str = "f0_Hz",
    half_window_sec: float = 0.05,
    distance: int = 10,
    prominence: float = 15.0,
) -> pl.DataFrame:
    """
    1) Detecta màxims i valls sobre la corba Savgol (cents).
    2) Per cada extrem, busca el màxim/mínim real a f0_Hz dins una finestra temporal.
    3) Desa també distance_h (sec) i distance_v (cents).

    Retorna 1 fila per extrem (events DF).
    """
    t = df[time_col].to_numpy()
    y = df[savgol_cents_col].to_numpy()
    raw = df[raw_hz_col].to_numpy()

    # màxims i valls (ignorem NaNs)
    y_max = np.where(np.isfinite(y), y, -np.inf)
    y_min = np.where(np.isfinite(y), -y, -np.inf)

    idx_max, _ = find_peaks(y_max, distance=distance, prominence=prominence)
    idx_min, _ = find_peaks(y_min, distance=distance, prominence=prominence)

    rows = []

    for kind, idxs in (("max", idx_max), ("min", idx_min)):
        for i in idxs:
            ts = float(t[i])
            ys = float(y[i])

            w = (np.abs(t - ts) <= half_window_sec) & np.isfinite(raw) & (raw > 0)
            if not np.any(w):
                continue

            j_all = np.where(w)[0]
            vals = raw[j_all]
            j = int(np.argmax(vals)) if kind == "max" else int(np.argmin(vals))
            ir = int(j_all[j])

            tr = float(t[ir])
            yr_hz = float(raw[ir])
            yr_cents = 1200.0 * np.log2(yr_hz / tonic_hz)

            rows.append({
                "recording_id": recording_id,
                "extremum_kind": kind,          # "max" o "min"
                "time_savgol": ts,
                "value_savgol_cents": ys,
                "time_raw": tr,
                "value_raw_hz": yr_hz,
                "value_raw_cents": yr_cents,
                "distance_h_sec": tr - ts,
                "distance_v_cents": yr_cents - ys,
            })

    if not rows:
        return pl.DataFrame([])

    return pl.DataFrame(rows).sort("time_savgol").with_row_index("extremum_id")
