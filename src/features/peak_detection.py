import polars as pl
import numpy as np

def find_all_peaks(
    df: pl.DataFrame,
    col: str,
    plateau_size: int = 2,
    neighborhood: int = 1,
    return_all_points: bool = True,
    include_edges: bool = False,
    gt_zero: bool = True
) -> np.ndarray:
    """
    Troba tots els índexs de màxims, mínims i plats de longitud ≤ `plateau_size`.
    Un plat és vàlid si els `neighborhood` punts abans i després són tots més alts (mínim) o més baixos (màxim).

    Args:
        df: DataFrame de Polars amb les dades.
        col: Nom de la columna a analitzar (ex: "f0_pchip").
        plateau_size: Mida màxima del plat a detectar (per defecte: 2).
        neighborhood: Nombre de punts veïns a comparar (per defecte: 1).
        return_all_points: Si `True`, retorna tots els punts del plat. Si `False`, retorna només els extrems.
        include_edges: Incloure els primers/últims punts.
        gt_zero: Només considerar valors positius.

    Returns:
        np.ndarray: Array amb els índexs dels *peaks* (màxims, mínims i plats).
    """
    values = df[col].to_numpy()
    peak_indices = set()

    i = 0
    while i < len(values):
        # Trobar la longitud del plat (1 ≤ longitud ≤ plateau_size)
        plateau_length = 1
        while (i + plateau_length < len(values) and
               plateau_length <= plateau_size and
               values[i] == values[i + plateau_length]):
            plateau_length += 1

        if plateau_length > 1:  # És un plat
            # Índexs del plat
            plateau_start = i
            plateau_end = i + plateau_length - 1

            # Comprovar si és un màxim o mínim
            left_idx = max(0, plateau_start - neighborhood)
            right_idx = min(len(values) - 1, plateau_end + neighborhood)

            # Valors abans i després del plat
            left_values = values[left_idx:plateau_start]
            right_values = values[plateau_end + 1:right_idx + 1]

            # Màxim plat: tots els punts del plat > tots els punts del veïnat
            if (all(v >= left_val for v in values[plateau_start:plateau_end + 1] for left_val in left_values) and
                all(v >= right_val for v in values[plateau_start:plateau_end + 1] for right_val in right_values)):
                if return_all_points:
                    peak_indices.update(range(plateau_start, plateau_end + 1))
                else:
                    peak_indices.update({plateau_start, plateau_end})

            # Mínim plat: tots els punts del plat < tots els punts del veïnat
            elif (all(v <= left_val for v in values[plateau_start:plateau_end + 1] for left_val in left_values) and
                  all(v <= right_val for v in values[plateau_start:plateau_end + 1] for right_val in right_values)):
                if return_all_points:
                    peak_indices.update(range(plateau_start, plateau_end + 1))
                else:
                    peak_indices.update({plateau_start, plateau_end})

            i += plateau_length  # Saltar el plat
        else:
            i += 1

    # Incloure els extrems si es demana
    if include_edges:
        if not np.isnan(values[0]) and (values[0] > 0 if gt_zero else True):
            peak_indices.add(0)
        if not np.isnan(values[-1]) and (values[-1] > 0 if gt_zero else True):
            peak_indices.add(len(values) - 1)

    return np.sort(list(peak_indices))





def extrema_to_dataframe(
    df: pl.DataFrame,
    extrema_idx: np.ndarray,
    col: str,
    extra_cols: list = None,
    plateau_size: int = 2,
    neighborhood: int = 1
) -> pl.DataFrame:
    """
    Crea un DataFrame de Polars amb el temps i els valors dels extrema.
    Opcionalment, inclou columnes addicionals (ex: "svara_mark").

    Args:
        df: DataFrame original de Polars.
        extrema_idx: Índexs dels extrema (de `find_all_peaks`).
        col: Nom de la columna amb els valors dels extrema.
        extra_cols: Llista de columnes addicionals per incloure.
        plateau_size: Mida del plat a detectar (per defecte: 2).
        neighborhood: Nombre de punts veïns a comparar (per defecte: 1).

    Returns:
        pl.DataFrame: DataFrame amb les dades dels extrema.
    """
    # Crea el DataFrame base amb el temps i els valors dels extrema
    extrema_df = pl.DataFrame({
        "time_rel_sec": df["time_rel_sec"][extrema_idx].to_numpy(),
        col: df[col][extrema_idx].to_numpy()
    })

    # Afegeix columnes addicionals si es demana
    if extra_cols:
        for c in extra_cols:
            if c in df.columns:
                extrema_df = extrema_df.with_columns(
                    pl.Series(c, df[c][extrema_idx].to_numpy())
                )

    return extrema_df





def add_peak_columns(
    df: pl.DataFrame,
    col: str,
    plateau_size: int = 2,
    neighborhood: int = 1,
    include_edges: bool = False,
    gt_zero: bool = True
) -> pl.DataFrame:
    
    """
    Afegeix columnes booleanes per marcar màxims i mínims (incloent-hi plats de mida `plateau_size`).

    Args:
        df: DataFrame de Polars.
        col: Nom de la columna (ex: "f0_pchip").
        plateau_size: Mida del plat a detectar (per defecte: 2).
        neighborhood: Nombre de punts veïns a comparar (per defecte: 1).
        include_edges: Incloure els primers/últims punts.
        gt_zero: Només considerar valors positius.

    Returns:
        pl.DataFrame: DataFrame amb les columnes `is_peak_max` i `is_peak_min`.
  """

    values = df[col].to_numpy()
    max_indices = set()
    min_indices = set()

    # Iterar per la sèrie per trobar plats de mida `plateau_size`
    i = 0
    while i < len(values) - plateau_size + 1:
        # Comprovar si hi ha un plat de mida `plateau_size`
        is_plateau = True
        for k in range(1, plateau_size):
            if values[i] != values[i + k]:
                is_plateau = False
                break

        if is_plateau:
            # Comprovar si és un màxim o mínim
            left_neighborhood = max(0, i - neighborhood)
            right_neighborhood = min(len(values), i + plateau_size + neighborhood)

            # Valors del veïnat
            left_values = values[left_neighborhood:i]
            right_values = values[i + plateau_size:right_neighborhood]

            # Màxim plat: tots els punts del plat > punts del veïnat
            if (all(v >= left_val for v in values[i:i+plateau_size] for left_val in left_values) and
                all(v >= right_val for v in values[i:i+plateau_size] for right_val in right_values)):
                max_indices.update(range(i, i + plateau_size))

            # Mínim plat: tots els punts del plat < punts del veïnat
            elif (all(v <= left_val for v in values[i:i+plateau_size] for left_val in left_values) and
                  all(v <= right_val for v in values[i:i+plateau_size] for right_val in right_values)):
                min_indices.update(range(i, i + plateau_size))

            i += plateau_size  # Saltar el plat
        else:
            i += 1

    # Incloure els extrems si es demana
    if include_edges:
        if not np.isnan(values[0]) and (values[0] > 0 if gt_zero else True):
            if values[0] >= values[1]:
                max_indices.add(0)
            else:
                min_indices.add(0)
        if not np.isnan(values[-1]) and (values[-1] > 0 if gt_zero else True):
            last_idx = len(values) - 1
            if values[-1] >= values[-2]:
                max_indices.add(last_idx)
            else:
                min_indices.add(last_idx)

    # Afegeix les columnes booleanes
    df = df.with_columns(
        pl.lit(False).alias("is_peak_max"),
        pl.lit(False).alias("is_peak_min")
    )

    # Marca els màxims
    df = df.with_columns(
        pl.when(pl.int_range(0, pl.count()).is_in(list(max_indices)))
          .then(True)
          .otherwise(pl.col("is_peak_max"))
          .alias("is_peak_max")
    )

    # Marca els mínims
    df = df.with_columns(
        pl.when(pl.int_range(0, pl.count()).is_in(list(min_indices)))
          .then(True)
          .otherwise(pl.col("is_peak_min"))
          .alias("is_peak_min")
    )

    return df
  