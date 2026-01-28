import settings as S

from src.io.pitch_io import load_preprocessed_pitch
from src.features.flatness import extract_flat_regions, add_flat_id
from src.io.pitch_io import save_flat_regions


def run_flat_regions(recording_id: str):
    tonic_hz = S.SARASUDA_TONICS.get(recording_id)

    df = load_preprocessed_pitch(
        recording_id=recording_id,
        root_dir=S.DATA_INTERIM,
        tonic_hz=tonic_hz,
        convert_to_cents=True,
    )

    # --- decide pitch col (cents) ---
    pitch_col = f"{S.PITCH_COL}_cents" if f"{S.PITCH_COL}_cents" in df.columns else S.PITCH_COL_CENTS

    # NOTE: no creem candidate aquí.
    # extract_flat_regions(...) el crearà si no existeix, via abs(deriv1) < threshold.

    df = extract_flat_regions(
        df,
        time_col=S.TIME_COL,
        pitch_col=pitch_col,
        pitch_unit="cents",
        candidate_col=S.CANDIDATE_COL,   # si no existeix, es crearà dins la funció
        out_col=S.STABLE_COL,
        min_duration_sec=S.MIN_STABLE_DURATION_SEC,
        cent_tolerance=S.TOLERANCE_CENTS,
        d1_threshold=S.D1_THRESHOLD,
        abs_deriv1_col=getattr(S, "ABS_DERIV1_COL", None),  # opcional per debug
        verbose=False,
    )

    df = add_flat_id(df, flat_col="flat_region", out_col="flat_id")

    out_path = save_flat_regions(df, recording_id=recording_id, root_dir=S.DATA_INTERIM)
    return out_path


if __name__ == "__main__":
    for recording_id in S.SARASUDA_VARNAM:
        print(f"Processing flat regions for recording: {recording_id}")
        out_path = run_flat_regions(recording_id)
        print(f"Saved flat regions to: {out_path}")
    #recording_id = S.CURRENT_PIECE
    #print(run_flat_regions(recording_id))
