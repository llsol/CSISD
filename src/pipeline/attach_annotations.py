from pathlib import Path
from src.io.annotation_io import load_annotations, save_annotations
from src.annotations.svara import attach_svara_annotations_to_pitch
from src.annotations.sections import attach_section_annotations_to_pitch
from src.io.pitch_io import load_pitch_file, save_pitch_file
from settings import SARASUDA_VARNAM
import polars as pl


def add_annotations_to_preprocessed_pitch(
    pitch_file_path: Path | str,
    svara_path: Path | str,
    section_path: Path | str,
    backup: bool = True,
    output_path: Path | str = None
) -> pl.DataFrame:
    """
    Obre el fitxer `preprocessed_pitch.parquet`, afegeix les 3 columnes,
    i desa el resultat al mateix fitxer (o a un de nou si s'especifica `output_path`).
    Si `backup=True`, crea una còpia de seguretat abans de modificar el fitxer original.

    Args:
        pitch_file_path: Path al fitxer `preprocessed_pitch.parquet`.
        svara_path: Path al fitxer TSV d'anotacions de svara.
        section_path: Path al fitxer TSV d'anotacions de secció.
        backup: Si True, crea una còpia de seguretat del fitxer original.
        output_path: Si es especifica, desa el resultat aquí. Si no, sobreescrivirà `pitch_file_path`.

    Returns:
        pl.DataFrame: DataFrame amb les 3 columnes afegides.
    """

    pitch_file_path = Path(pitch_file_path)

    # 1. Còpia de seguretat (opcional)
    if backup:
        backup_path = pitch_file_path.with_stem(f"{pitch_file_path.stem}_backup")
        pl.read_parquet(pitch_file_path).write_parquet(backup_path)
        print(f"Còpia de seguretat creada: {backup_path}")

    # 2. Carrega el fitxer preprocessat (amb TOTES les seves columnes originals)
    df_pitch = load_pitch_file(pitch_file_path)

    # 3. Carrega les anotacions
    df_svaras = load_annotations(svara_path, "svara")
    df_sections = load_annotations(section_path, "section")

    # 4. Afegeix les 3 columnes utilitzant els TEUS mètodes originals
    df_pitch = attach_svara_annotations_to_pitch(df_pitch, df_svaras)
    df_pitch = attach_section_annotations_to_pitch(df_pitch, df_sections)

    # 5. Desa el resultat (al mateix fitxer o a un de nou)
    save_path = output_path if output_path else pitch_file_path
    save_pitch_file(df_pitch, save_path, engine='polars')
    print(f"Fitxer actualitzat: {save_path}")

    return df_pitch

if __name__ == "__main__":
    corpus_root = Path("data/corpus")
    for recording_id in SARASUDA_VARNAM:
        pitch_file_path = Path("data/interim") / recording_id / "pitch"/ f"{recording_id}_pitch_preprocessed.parquet"
        svara_path = corpus_root / recording_id / "raw" / f"{recording_id}_ann_svara.tsv"
        section_path = corpus_root / recording_id / "annotations" / f"{recording_id}_ann_section.tsv"
        add_annotations_to_preprocessed_pitch(pitch_file_path=pitch_file_path, svara_path=svara_path, section_path=section_path)