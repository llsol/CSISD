"""
FTA-Net pitch extraction on a Carnatic vocal recording.

Uses compiam.load_model("melody:ftanet-carnatic") — trained on Carnatic vocals.
Requires tensorflow in the compiam environment.

Run with the compiam environment:
    # Original recording (MP3 from corpus):
    /home/lluis/anaconda3/envs/compiam/bin/python -m src.source_separation.ftanet_predict srs_v1_svd_sav

    # U-Net separated voice:
    /home/lluis/anaconda3/envs/compiam/bin/python -m src.source_separation.ftanet_predict srs_v1_svd_sav --unet

Writes : data/interim/{recording_id}/pitch_raw/{recording_id}_{suffix}_ftanet_raw.npy
         shape: (N, 2) — col 0: time_sec, col 1: f0_Hz (0.0 = unvoiced)

  suffix = "original"  when reading from corpus MP3
  suffix = "unet"      when reading from U-Net separated voice
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
MODEL_SR   = 8000


def predict_pitch(recording_id: str, audio_path: Path, suffix: str) -> Path:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    out_dir  = settings.DATA_INTERIM / recording_id / "pitch_raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{recording_id}_{suffix}_ftanet_raw.npy"

    print(f"[ftanet] loading {audio_path} → resample to {MODEL_SR} Hz")
    audio, _ = librosa.load(audio_path, sr=MODEL_SR, mono=True)

    print(f"[ftanet] loading {MODEL_NAME} ...")
    model = compiam.load_model(MODEL_NAME)

    print(f"[ftanet] predicting ({len(audio)/MODEL_SR:.1f}s) ...")
    pitch = model.predict(audio, input_sr=MODEL_SR)  # (N, 2): [time_sec, f0_Hz]

    pitch = np.asarray(pitch, dtype=np.float64)
    np.save(out_path, pitch)
    print(f"[ftanet] {len(pitch)} frames → {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="FTA-Net pitch extraction on Carnatic vocal audio."
    )
    parser.add_argument(
        "recording_id",
        nargs="?",
        default=settings.CURRENT_PIECE,
        help=f"Recording ID (default: {settings.CURRENT_PIECE})",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--unet", action="store_true",
        help="Use U-Net separated voice.",
    )
    source.add_argument(
        "--as", dest="as_model", action="store_true",
        help="Use BS-RoFormer (audio-separator) separated voice.",
    )
    args = parser.parse_args()

    if args.unet:
        audio_path = (
            settings.DATA_INTERIM / "source_separation" / "separated"
            / args.recording_id / f"{args.recording_id}_unet_voice.wav"
        )
        suffix = "unet"
    elif args.as_model:
        audio_path = (
            settings.DATA_INTERIM / "source_separation_as" / "separated"
            / args.recording_id / f"{args.recording_id}_as_voice.wav"
        )
        suffix = "as"
    else:
        audio_path = (
            settings.DATA_CORPUS / args.recording_id / "audio"
            / f"{args.recording_id}.mp3"
        )
        suffix = "original"

    predict_pitch(args.recording_id, audio_path, suffix)


if __name__ == "__main__":
    main()
