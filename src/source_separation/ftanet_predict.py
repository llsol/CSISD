"""
FTA-Net pitch extraction on tampura-attenuated audio.

Uses compiam.load_model("melody:ftanet-carnatic") — trained on Carnatic vocals.
Requires tensorflow in the compiam environment.

Run with the compiam environment:
    /home/lluis/anaconda3/envs/compiam/bin/python -m src.source_separation.ftanet_predict <recording_id>

Reads  : data/interim/{recording_id}/audio_notampura/{recording_id}_clean.wav
Writes : data/interim/{recording_id}/pitch_raw/{recording_id}_ftanet_raw.npy
         shape: (N, 2) — col 0: time_sec, col 1: f0_Hz (0.0 = unvoiced)

Load in main env with:
    import numpy as np
    pitch = np.load("...ftanet_raw.npy")
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import librosa
import compiam

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import settings

MODEL_NAME = "melody:ftanet-carnatic"
MODEL_SR = 8000


def predict_pitch(recording_id: str) -> Path:
#    audio_path = (
#        settings.DATA_INTERIM
#        / recording_id
#        / "audio_notampura"
#        / f"{recording_id}_clean.wav"
#    )
    audio_path = (
        settings.DATA_CORPUS
        / recording_id
        / "audio"
        / f"{recording_id}.mp3"
    )
    if not audio_path.exists():
        raise FileNotFoundError(
            f"Clean audio not found: {audio_path}\n"
            "Run the tampura separation first."
        )

    out_dir = settings.DATA_INTERIM / recording_id / "pitch_raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{recording_id}_reproduction_ftanet_raw.npy"

    print(f"[ftanet] loading {audio_path} → resample to {MODEL_SR} Hz")
    audio, _ = librosa.load(audio_path, sr=MODEL_SR, mono=True)

    print(f"[ftanet] loading {MODEL_NAME} ...")
    model = compiam.load_model(MODEL_NAME)  # handles download + weights automatically

    print(f"[ftanet] predicting ({len(audio)/MODEL_SR:.1f}s) ...")
    pitch = model.predict(audio, input_sr=MODEL_SR)  # (N, 2): [time_sec, f0_Hz]

    pitch = np.asarray(pitch, dtype=np.float64)
    np.save(out_path, pitch)
    print(f"[ftanet] {len(pitch)} frames → {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="FTA-ResNet pitch extraction on tampura-attenuated audio."
    )
    parser.add_argument(
        "recording_id",
        nargs="?",
        default=settings.CURRENT_PIECE,
        help=f"Recording ID (default: {settings.CURRENT_PIECE})",
    )
    args = parser.parse_args()
    predict_pitch(args.recording_id)


if __name__ == "__main__":
    main()
