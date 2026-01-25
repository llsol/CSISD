
import numpy as np
from scipy.signal import savgol_filter
import polars as pl


def compute_derivatives(
    df: pl.DataFrame,
    col_in: str = "f0_pchip",
    window_length: int = 13,
    polyorder: int = 3,
):
    """
    Derivades 1a i 2a amb Savitzky-Golay per grups (group_id).
    Retorna dict amb arrays de longitud df.height:
      {"d1": np.ndarray(N,), "d2": np.ndarray(N,)}
    """

    n = df.height

    # valida columnes
    if col_in not in df.columns:
        raise ValueError(f"df no té la columna '{col_in}'")
    if "group_id" not in df.columns:
        raise ValueError("df no té la columna 'group_id'")

    # window_length ha de ser >= polyorder+2 i normalment senar
    if window_length < polyorder + 2:
        window_length = polyorder + 2
    if window_length % 2 == 0:
        window_length += 1

    f = df[col_in].to_numpy()
    group = df["group_id"].to_numpy()

    groups = np.unique(group)

    d1_savgol = np.full(n, np.nan, dtype=float)
    d2_savgol = np.full(n, np.nan, dtype=float)

    for g in groups:
        mask = (group == g)
        idx = np.where(mask)[0]

        if len(idx) < window_length:
            continue

        f_group = f[idx]

        not_nan = np.isfinite(f_group)
        if not np.any(not_nan):
            continue

        first = int(np.argmax(not_nan))
        last = int(len(f_group) - 1 - np.argmax(not_nan[::-1]))

        f_trim = f_group[first:last + 1]

        # si hi ha NaNs dins el tram útil, no filtrem (mateix criteri que tenies)
        if np.any(np.isnan(f_trim)):
            continue

        if len(f_trim) < window_length:
            continue

        try:
            f_trim_d1 = savgol_filter(
                f_trim,
                window_length=window_length,
                polyorder=polyorder,
                deriv=1,
                mode="interp",
            )
            f_trim_d2 = savgol_filter(
                f_trim,
                window_length=window_length,
                polyorder=polyorder,
                deriv=2,
                mode="interp",
            )
        except Exception:
            # no petem el pipeline: simplement deixem NaNs en aquest grup
            # print(f"Could not apply Savitzky-Golay filter to group {g}.")
            continue

        # col·loca resultats al seu lloc dins el grup
        d1_savgol[idx[first:last + 1]] = f_trim_d1
        d2_savgol[idx[first:last + 1]] = f_trim_d2

    return {"d1": d1_savgol, "d2": d2_savgol}



def compute_derivative_features(
    df: pl.DataFrame,
    col_d1: str = "d1",
    col_d2: str = "d2",
):
    """
    Features estadístiques de derivades dins un segment.
    Espera que df tingui columnes col_d1 i col_d2 (p.ex. d1/d2).
    Retorna dict d'escalars.
    """
    if col_d1 not in df.columns or col_d2 not in df.columns:
        raise ValueError(f"Falten columnes: calen '{col_d1}' i '{col_d2}'")

    d1 = df[col_d1].to_numpy()
    d2 = df[col_d2].to_numpy()

    def stats(x):
        x = x[np.isfinite(x)]
        if len(x) == 0:
            return (np.nan, np.nan, np.nan, np.nan)
        return (
            float(np.mean(x)),
            float(np.std(x)),
            float(np.mean(np.abs(x))),
            float(np.std(np.abs(x))),
        )

    d1_mean, d1_std, d1_abs_mean, d1_abs_std = stats(d1)
    d2_mean, d2_std, d2_abs_mean, d2_abs_std = stats(d2)

    return {
        "d1_mean": d1_mean,
        "d1_std": d1_std,
        "d1_abs_mean": d1_abs_mean,
        "d1_abs_std": d1_abs_std,
        "d2_mean": d2_mean,
        "d2_std": d2_std,
        "d2_abs_mean": d2_abs_mean,
        "d2_abs_std": d2_abs_std,
    }