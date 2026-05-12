"""
FTA-Net pitch extraction on a Carnatic vocal recording.

Uses compiam.load_model("melody:ftanet-carnatic") — trained on Carnatic vocals.
Requires tensorflow in the compiam environment.

Run with the compiam environment:
    # Original recording (MP3 from corpus):
    /home/lluis/anaconda3/envs/compiam/bin/python -m src.pitch_extraction.ftanet_predict srs_v1_svd_sav

    # U-Net separated voice:
    /home/lluis/anaconda3/envs/compiam/bin/python -m src.pitch_extraction.ftanet_predict srs_v1_svd_sav --unet

    # BS-RoFormer (audio-separator) separated voice:
    /home/lluis/anaconda3/envs/compiam/bin/python -m src.pitch_extraction.ftanet_predict srs_v1_svd_sav --as

    # All SCMS clips (original audio):
    /home/lluis/anaconda3/envs/compiam/bin/python -m src.pitch_extraction.ftanet_predict --scms

    # All SCMS clips (BS-RoFormer separated voice):
    /home/lluis/anaconda3/envs/compiam/bin/python -m src.pitch_extraction.ftanet_predict --scms --as

    # SCMS test split only, skip already-done clips:
    /home/lluis/anaconda3/envs/compiam/bin/python -m src.pitch_extraction.ftanet_predict --scms --split test --skip-existing

Corpus output : data/interim/cv_pitch_ftanet/{suffix}/{recording_id}/{recording_id}_{suffix}_ftanet_raw.npy
SCMS output   : data/interim/scms_pitch_ftanet/{suffix}/{fragment}_{suffix}_ftanet_raw.npy
Shape: (N, 2) — col 0: time_sec, col 1: f0_Hz (0.0 = unvoiced)

  suffix = "original"  when reading from corpus MP3 or SCMS original audio
  suffix = "unet"      when reading from U-Net separated voice (corpus only)
  suffix = "as"        when reading from BS-RoFormer separated voice
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import librosa
import compiam

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import settings
from src.pitch_extraction.swiftf0_finetune.dataset import scms_official_split

MODEL_NAME        = "melody:ftanet-carnatic"
MODEL_SR          = 8000
SCMS_AUDIO_DIR    = settings.PROJECT_ROOT / "data" / "datasets" / "scms" / "audio"
SCMS_SEPARATED_DIR = settings.DATA_INTERIM / "scms_separated"
SCMS_PITCH_DIR    = settings.DATA_INTERIM / "scms_pitch_ftanet"


def predict_pitch(
    recording_id: str,
    audio_path: Path,
    suffix: str,
    model=None,
    out_path: Optional[Path] = None,
) -> Path:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    if out_path is None:
        out_dir = settings.DATA_INTERIM / "cv_pitch_ftanet" / suffix / recording_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{recording_id}_{suffix}_ftanet_raw.npy"

    print(f"[ftanet] loading {audio_path} → resample to {MODEL_SR} Hz")
    audio, _ = librosa.load(audio_path, sr=MODEL_SR, mono=True)

    if model is None:
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
        default=None,
        help=f"Recording ID (default: {settings.CURRENT_PIECE})",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Process all recordings in settings.SARASUDA_VARNAM.",
    )
    parser.add_argument(
        "--scms", action="store_true",
        help="Process all SCMS clips instead of a corpus recording.",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip clips whose output .npy already exists (useful with --scms).",
    )
    parser.add_argument(
        "--split", choices=["train", "test", "all"], default="all",
        help="SCMS split to process (default: all).",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--unet", action="store_true",
        help="Use U-Net separated voice (corpus mode only).",
    )
    source.add_argument(
        "--as", dest="as_model", action="store_true",
        help="Use BS-RoFormer (audio-separator) separated voice.",
    )
    args = parser.parse_args()

    # ── SCMS mode ──────────────────────────────────────────────────────────────
    if args.scms:
        train_stems, test_stems = scms_official_split(
            settings.PROJECT_ROOT / "data" / "datasets" / "scms"
        )
        if args.split == "train":
            stems = train_stems
        elif args.split == "test":
            stems = test_stems
        else:
            stems = train_stems + test_stems

        suffix = "as" if args.as_model else "original"
        n_total = len(stems)
        n_done  = 0

        print(f"[ftanet] loading {MODEL_NAME} ...")
        model = compiam.load_model(MODEL_NAME)
        print(f"[ftanet] SCMS mode: {n_total} clips  suffix={suffix} → {SCMS_PITCH_DIR}")

        out_subdir = SCMS_PITCH_DIR / suffix
        out_subdir.mkdir(parents=True, exist_ok=True)
        n_missing = 0
        for i, stem in enumerate(stems):
            out_path = out_subdir / f"{stem}_{suffix}_ftanet_raw.npy"
            if args.skip_existing and out_path.exists():
                continue

            if args.as_model:
                audio_path = SCMS_SEPARATED_DIR / stem / f"{stem}_as_voice.wav"
            else:
                audio_path = SCMS_AUDIO_DIR / f"{stem}.wav"

            if not audio_path.exists():
                print(f"  [{i+1}/{n_total}] {stem}  [skip — audio not found]")
                n_missing += 1
                continue

            print(f"\n── [{i+1}/{n_total}] {stem} ──")
            predict_pitch("_scms", audio_path, suffix, model=model, out_path=out_path)
            n_done += 1

        print(f"\n[ftanet] SCMS done: {n_done} processed, {n_missing} missing audio, "
              f"{n_total - n_done - n_missing} skipped.")
        return

    # ── corpus mode ────────────────────────────────────────────────────────────
    if args.all:
        recordings = settings.SARASUDA_VARNAM
    else:
        recordings = [args.recording_id or settings.CURRENT_PIECE]

    print(f"[ftanet] loading {MODEL_NAME} ...")
    model = compiam.load_model(MODEL_NAME)

    for recording_id in recordings:
        if args.unet:
            audio_path = (
                settings.DATA_INTERIM / "source_separation" / "separated"
                / recording_id / f"{recording_id}_unet_voice.wav"
            )
            suffix = "unet"
        elif args.as_model:
            audio_path = (
                settings.DATA_INTERIM / "source_separation_as" / "separated"
                / recording_id / f"{recording_id}_as_voice.wav"
            )
            suffix = "as"
        else:
            audio_path = (
                settings.DATA_CORPUS / recording_id / "audio"
                / f"{recording_id}.mp3"
            )
            suffix = "original"

        print(f"\n── {recording_id}  [{suffix}] ──")
        predict_pitch(recording_id, audio_path, suffix, model=model)


if __name__ == "__main__":
    main()
