from pathlib import Path

from src.annotations.svara import (
    load_svara_annotations,
    save_svara_annotations_parquet,
)

from src.annotations.sections import (
    load_section_annotations,
    save_section_annotations_parquet,
)


def update_annotation_parquets_for_recording(
    recording_id: str,
    corpus_root: Path | str = "data/corpus",
    processed_root: Path | str = "data/processed/annotations",
):
    """
    Create/update annotation parquets (svara + sections) for one recording.

    Reads from:
        data/corpus/<recording_id>/annotations/

    Writes to:
        data/processed/annotations/<recording_id>/
    """

    corpus_root = Path(corpus_root)
    processed_root = Path(processed_root)

    svara_ann_dir = corpus_root / recording_id / "raw"
    section_ann_dir = corpus_root / recording_id / "annotations" 
    if not svara_ann_dir.exists():
        raise FileNotFoundError(f"No annotations dir for {recording_id}")

    if not section_ann_dir.exists():
        raise FileNotFoundError(f"No annotations dir for {recording_id}")
    
    out_dir = processed_root / recording_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- SVARA ----------
    for tsv in svara_ann_dir.glob("*_ann_svara*.tsv"):
        df_svara = load_svara_annotations(tsv, engine="polars")
        out_path = out_dir / "svara_segments.parquet"
        save_svara_annotations_parquet(df_svara, out_path)

    # ---------- SECTIONS ----------
    for tsv in section_ann_dir.glob("*_ann_section*.tsv"):
        out_path = out_dir / "sections.parquet"
        save_section_annotations_parquet(tsv, out_path, engine="polars")





def update_all_annotation_parquets(
    corpus_root: Path | str = "data/corpus",
    processed_root: Path | str = "data/processed/annotations",
):
    corpus_root = Path(corpus_root)

    for rec_dir in corpus_root.iterdir():
        if not rec_dir.is_dir():
            continue

        recording_id = rec_dir.name
        ann_dir = rec_dir / "annotations"
        if not ann_dir.exists():
            continue

        update_annotation_parquets_for_recording(
            recording_id,
            corpus_root=corpus_root,
            processed_root=processed_root,
        )