from src.io.pitch_io import load_preprocessed_pitch, save_peaks
from src.features.peaks import extract_peaks
import settings as S

RECORDING_IDS = S.SARASUDA_VARNAM
TONICS = S.SARASUDA_TONICS

if __name__ == "__main__":

    print(f"Processing {len(RECORDING_IDS)} recordings:")
    print(RECORDING_IDS)
    print("-" * 40)

    for recording_id in RECORDING_IDS:
        tonic_hz = TONICS[recording_id]

        print(f"[START] recording_id = {recording_id} | tonic = {tonic_hz:.2f} Hz")

        # --- load pitch ---
        df_pitch = load_preprocessed_pitch(
            recording_id,
            tonic_hz=tonic_hz,
            convert_to_cents=True,
        )
        print(f"  Loaded pitch: {df_pitch.height} frames")

        # --- extract peaks ---
        df_peaks = extract_peaks(
            df_pitch,
            recording_id=recording_id,
            tonic_hz=tonic_hz,
            
        )
        print(f"  Extracted extrema: {df_peaks.height} events")

        # --- save ---
        save_peaks(df_peaks, recording_id)
        print(f"[DONE]  recording_id = {recording_id}")
        print("-" * 40)
