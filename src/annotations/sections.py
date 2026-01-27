import polars as pl
import numpy as np

def attach_section_annotations_to_pitch(
    df_pitch: pl.DataFrame,
    df_sections: pl.DataFrame,
    threshold_sec: float = 1.0
) -> pl.DataFrame:
    """
    Assigna seccions al pitch.

    - df_sections té només punts d'inici: (start_time_sec, section_label)
    - Considerem que una secció acaba quan comença la següent.
    - Si hi ha svara_label i hi ha un *_start prou proper, "snap" al frame del svara_start.
    - Output:
        - section_label: propagat a totes les files dins la secció (nullable si fora de seccions)
        - section_id: int (nullable si fora de seccions)
    """
    df_pitch = df_pitch.clone().with_row_index("row_nr")

    if "time_rel_sec" not in df_pitch.columns:
        raise ValueError("df_pitch no té columna 'time_rel_sec'")
    if "start_time_sec" not in df_sections.columns or "section_label" not in df_sections.columns:
        raise ValueError("df_sections ha de tenir 'start_time_sec' i 'section_label'")

    # --- CONTEXT: dades base ---
    t = df_pitch.get_column("time_rel_sec")
    n = df_pitch.height
    pitch_times = t.to_numpy()

    # Extreu temps de svara_start si existeixen (per fer snap)
    if "svara_label" in df_pitch.columns:
        svara_start_times = (
            df_pitch
            .filter(pl.col("svara_label").fill_null("").str.ends_with("_start"))
            .get_column("time_rel_sec")
            .to_numpy()
        )
    else:
        svara_start_times = np.array([], dtype=float)

    # --- CONTEXT: normalitza/ordena seccions ---
    df_sections = df_sections.sort("start_time_sec")

    # --- CONTEXT AFEGIT: calcula row_nr d'inici per cada secció (amb snap opcional) ---
    section_starts = []
    section_labels = df_sections.get_column("section_label").to_list()

    for t_sec in df_sections.get_column("start_time_sec").to_list():
        # decideix a quin temps fem match (snap a svara_start si és proper)
        if svara_start_times.size > 0:
            nearest_idx = int(np.abs(svara_start_times - t_sec).argmin())
            if abs(float(svara_start_times[nearest_idx]) - float(t_sec)) <= threshold_sec:
                target_t = float(svara_start_times[nearest_idx])
            else:
                target_t = float(t_sec)
        else:
            target_t = float(t_sec)

        # passa de temps -> índex de fila (primer punt >= target_t)
        idx = int(np.searchsorted(pitch_times, target_t, side="left"))
        section_starts.append(idx)

    section_starts = np.asarray(section_starts, dtype=int)

    # filtra starts que cauen fora del pitch
    valid = (section_starts < n)
    section_starts = section_starts[valid]
    section_labels = [lab for lab, ok in zip(section_labels, valid) if ok]

    if len(section_starts) == 0:
        # no hi ha seccions aplicables: retorna sense afegir res
        return df_pitch.drop("row_nr")

    # elimina duplicats de start (si n'hi ha) mantenint el primer (ordre ja sorted)
    keep = []
    last = None
    for i, s in enumerate(section_starts):
        if last is None or s != last:
            keep.append(i)
            last = s
    section_starts = section_starts[keep]
    section_labels = [section_labels[i] for i in keep]

    # --- CONTEXT AFEGIT: construcció de rangs start..(next_start-1), o fins final ---
    section_ends = np.r_[section_starts[1:] - 1, n - 1]

    df_ranges = pl.DataFrame({
        "_start": section_starts,
        "_end": section_ends,
        "section_label": section_labels,
    }).with_row_index("section_id")

    df_section_ranges = (
        df_ranges
        .filter(pl.col("_start") <= pl.col("_end"))
        .with_columns(
            pl.int_ranges(
                pl.col("_start").cast(pl.Int64),
                (pl.col("_end") + 1).cast(pl.Int64)
            ).alias("row_nr")
        )
        .select(["row_nr", "section_id", "section_label"])
        .explode("row_nr")
    )

    df_section_map = (
        df_section_ranges
        .group_by("row_nr")
        .agg([
            pl.col("section_id").min().alias("section_id"),
            pl.col("section_label").first().alias("section_label"),
        ])
    )

    # neteja columnes conflictives prèvies
    df_pitch = df_pitch.drop(
        ["section_id", "section_id_right", "section_label_right", "section_label"],
        strict=False
    )

    # join i neteja
    df_pitch = (
        df_pitch
        .join(df_section_map, on="row_nr", how="left")
        .drop("row_nr")
    )

    return df_pitch


"""
def attach_section_annotations_to_pitch(
    df_pitch: pl.DataFrame,
    df_sections: pl.DataFrame,
    threshold_sec: float = 1.0
) -> pl.DataFrame:
    '''
    Assigna etiquetes de secció al DataFrame de pitch.
    Usa les marques de svara si estan disponibles i són properes.
    '''
    df_pitch = df_pitch.clone().with_row_index("row_nr")

    pitch_times = df_pitch["time_rel_sec"].to_numpy()

    # Extreu temps de svara_start si existeixen
    svara_start_times = (
        df_pitch
        .filter(pl.col("svara_label").fill_null("").str.ends_with("_start"))
        ["time_rel_sec"]
        .to_numpy()
    )

    # Inicialitza columna (nullable string)
    df_pitch = df_pitch.with_columns(pl.lit(None).cast(pl.Utf8).alias("section_label"))

    for t_sec, label in df_sections[["start_time_sec", "section_label"]].rows():
        if svara_start_times.size > 0:
            nearest_idx = np.abs(svara_start_times - t_sec).argmin()
            if abs(svara_start_times[nearest_idx] - t_sec) <= threshold_sec:
                idx_pitch = np.abs(pitch_times - svara_start_times[nearest_idx]).argmin()
            else:
                idx_pitch = np.abs(pitch_times - t_sec).argmin()
        else:
            idx_pitch = np.abs(pitch_times - t_sec).argmin()

        # Assigna etiqueta per índex (no per float equality)
        df_pitch = df_pitch.with_columns(
            pl.when(pl.col("row_nr") == int(idx_pitch))
              .then(pl.lit(label))
              .otherwise(pl.col("section_label"))
              .alias("section_label")
        )

    return df_pitch.drop("row_nr")
"""