"""
SwiftF0 pitch extraction on Carnatic vocal recordings.

SwiftF0: lightweight CNN (~96K params), 16ms frames, 42× faster than CREPE.
Install:  pip install swift-f0
Requires: PyTorch environment (e.g. bss or compiam).

Usage:
    # Original corpus audio (MP3):
    python -m src.pitch_extraction.swiftf0_predict srs_v1_svd_sav

    # BS-RoFormer separated voice:
    python -m src.pitch_extraction.swiftf0_predict srs_v1_svd_sav --as

    # All SARASUDA_VARNAM:
    python -m src.pitch_extraction.swiftf0_predict --all
    python -m src.pitch_extraction.swiftf0_predict --all --as

    # All SCMS clips (original audio):
    python -m src.pitch_extraction.swiftf0_predict --scms

    # All SCMS clips (BS-RoFormer separated):
    python -m src.pitch_extraction.swiftf0_predict --scms --as

    # SCMS test split only, skip already-done:
    python -m src.pitch_extraction.swiftf0_predict --scms --split test --skip-existing

CV output  : data/interim/cv_pitch_swiftf0/{suffix}/{id}/{id}_{suffix}_swiftf0_raw.npy
SCMS output: data/interim/scms_pitch_swiftf0/{suffix}/{stem}_{suffix}_swiftf0_raw.npy
Shape: (N, 2) — col 0: time_sec, col 1: f0_Hz (0.0 = unvoiced)
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import settings
from src.pitch_extraction.swiftf0_finetune.dataset import scms_official_split

CONFIDENCE_THRESHOLD = 0.9
FMIN                 = 46.875
FMAX                 = 2093.75
MODEL_SR             = 16000
SCMS_AUDIO_DIR       = settings.PROJECT_ROOT / "data" / "datasets" / "scms" / "audio"
SCMS_SEPARATED_DIR   = settings.INTERIM_SEP_SCMS
SCMS_PITCH_DIR       = settings.INTERIM_PITCH_SCMS / "swiftf0"


def _find_corpus_audio(recording_id: str) -> Path:
    audio_dir = settings.DATA_CORPUS / recording_id / "audio"
    for ext in ("mp3", "wav", "flac", "ogg"):
        candidates = list(audio_dir.glob(f"*.{ext}"))
        if candidates:
            return candidates[0]
    raise FileNotFoundError(f"No audio file found in {audio_dir}")


def _run_swiftf0(audio_path: Path) -> tuple[np.ndarray, np.ndarray]:
    import librosa
    from swift_f0 import SwiftF0

    audio, _ = librosa.load(str(audio_path), sr=MODEL_SR, mono=True)
    detector = SwiftF0(fmin=FMIN, fmax=FMAX,
                       confidence_threshold=CONFIDENCE_THRESHOLD)
    result   = detector.detect_from_array(audio, sample_rate=MODEL_SR)

    f0_hz  = np.asarray(result.pitch_hz,   dtype=np.float64)
    times  = np.asarray(result.timestamps, dtype=np.float64)
    voiced = np.asarray(result.voicing,    dtype=bool)
    f0_hz[~voiced] = 0.0
    return times, f0_hz


def predict_pitch(audio_path: Path, out_path: Path) -> Path:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    print(f"[swiftf0] loading {audio_path.name} → resample to {MODEL_SR} Hz")
    times, f0_hz = _run_swiftf0(audio_path)

    pitch = np.column_stack([times, f0_hz])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, pitch)

    n_voiced = (f0_hz > 0).sum()
    print(
        f"[swiftf0] {len(pitch):,} frames  "
        f"({times[-1]:.1f} s)  "
        f"voiced: {n_voiced:,} ({n_voiced/len(pitch)*100:.1f}%)"
    )
    print(f"[swiftf0] → {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="SwiftF0 (original weights) pitch extraction."
    )
    parser.add_argument(
        "recordings", nargs="*",
        help="Recording ID(s). Defaults to CURRENT_PIECE; use --all for SARASUDA_VARNAM.",
    )
    parser.add_argument("--all",  action="store_true",
                        help="Process all recordings in settings.SARASUDA_VARNAM.")
    parser.add_argument("--scms", action="store_true",
                        help="Process all SCMS clips.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip clips whose output .npy already exists (--scms mode).")
    parser.add_argument("--split", choices=["train", "test", "all"], default="all",
                        help="SCMS split to process (default: all).")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--unet", action="store_true",
                        help="Use U-Net separated voice (CV mode only).")
    source.add_argument("--as", dest="as_model", action="store_true",
                        help="Use BS-RoFormer separated voice.")
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

        suffix     = "as" if args.as_model else "original"
        out_subdir = SCMS_PITCH_DIR / suffix
        out_subdir.mkdir(parents=True, exist_ok=True)
        n_total, n_done = len(stems), 0
        print(f"[swiftf0] SCMS mode: {n_total} clips  suffix={suffix} → {out_subdir}")

        for i, stem in enumerate(stems):
            out_path = out_subdir / f"{stem}_{suffix}_swiftf0_raw.npy"
            if args.skip_existing and out_path.exists():
                continue
            audio_path = (SCMS_SEPARATED_DIR / stem / f"{stem}_as_voice.wav"
                          if args.as_model else SCMS_AUDIO_DIR / f"{stem}.wav")
            print(f"\n── [{i+1}/{n_total}] {stem} ──")
            predict_pitch(audio_path, out_path)
            n_done += 1

        print(f"\n[swiftf0] SCMS done: {n_done} processed, {n_total - n_done} skipped.")
        return

    # ── corpus mode ────────────────────────────────────────────────────────────
    if args.all:
        recordings = settings.SARASUDA_VARNAM
    elif args.recordings:
        recordings = args.recordings
    else:
        recordings = [settings.CURRENT_PIECE]

    for recording_id in recordings:
        if args.unet:
            audio_path = (
                settings.INTERIM_SEP_CV_UNET
                / recording_id / f"{recording_id}_unet_voice.wav"
            )
            suffix = "unet"
        elif args.as_model:
            audio_path = (
                settings.INTERIM_SEP_CV_AS
                / recording_id / f"{recording_id}_as_voice.wav"
            )
            suffix = "as"
        else:
            audio_path = _find_corpus_audio(recording_id)
            suffix = "original"

        out_path = (settings.INTERIM_PITCH_CV / "swiftf0"
                    / suffix / recording_id
                    / f"{recording_id}_{suffix}_swiftf0_raw.npy")
        print(f"\n── {recording_id}  [{suffix}] ──")
        predict_pitch(audio_path, out_path)


if __name__ == "__main__":
    main()
