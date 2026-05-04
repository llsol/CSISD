"""
SwiftF0 pitch extraction on Carnatic vocal recordings.

SwiftF0: lightweight CNN (~96K params), 16ms frames, 42× faster than CREPE.
Install:  pip install swift-f0
Requires: PyTorch environment (e.g. bss or compiam).

Usage:
    # Original corpus audio (MP3):
    python -m src.pitch_extraction.swiftf0_predict srs_v1_svd_sav

    # U-Net separated voice:
    python -m src.pitch_extraction.swiftf0_predict srs_v1_svd_sav --unet

    # BS-RoFormer separated voice:
    python -m src.pitch_extraction.swiftf0_predict srs_v1_svd_sav --as

    # All SARASUDA_VARNAM with U-Net separated voice:
    python -m src.pitch_extraction.swiftf0_predict --all --unet

Reads audio from:
    Original : data/corpus/{id}/audio/{id}.mp3
    U-Net    : data/interim/source_separation/separated/{id}/{id}_unet_voice.wav
    BS-Roform: data/interim/source_separation_as/separated/{id}/{id}_as_voice.wav

Writes:
    data/interim/{id}/pitch_raw/{id}_{suffix}_swiftf0_raw.npy
    shape: (N, 2) — col 0: time_sec, col 1: f0_Hz (0.0 = unvoiced)

Suffix mapping:
    (no flag)  → "original"
    --unet     → "unet"
    --as       → "as"
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import settings

CONFIDENCE_THRESHOLD = 0.9   # voicing threshold (paper default)
FMIN = 46.875                # Hz — lower bound (matches SwiftF0 training range)
FMAX = 2093.75               # Hz — upper bound
MODEL_SR = 16000             # SwiftF0 native sample rate


def _find_corpus_audio(recording_id: str) -> Path:
    audio_dir = settings.DATA_CORPUS / recording_id / "audio"
    for ext in ("mp3", "wav", "flac", "ogg"):
        candidates = list(audio_dir.glob(f"*.{ext}"))
        if candidates:
            return candidates[0]
    raise FileNotFoundError(f"No audio file found in {audio_dir}")


def predict_pitch(recording_id: str, audio_path: Path, suffix: str) -> Path:
    import librosa
    from swift_f0 import SwiftF0

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    out_dir  = settings.DATA_INTERIM / recording_id / "pitch_raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{recording_id}_{suffix}_swiftf0_raw.npy"

    print(f"[swiftf0] loading {audio_path.name} → resample to {MODEL_SR} Hz")
    audio, _ = librosa.load(str(audio_path), sr=MODEL_SR, mono=True)

    detector = SwiftF0(
        fmin=FMIN,
        fmax=FMAX,
        confidence_threshold=CONFIDENCE_THRESHOLD,
    )

    print(f"[swiftf0] predicting ({len(audio)/MODEL_SR:.1f} s) ...")
    result = detector.detect_from_array(audio, sample_rate=MODEL_SR)

    # Convert to (N, 2) array matching FTA-Net convention: 0 Hz = unvoiced
    f0_hz  = np.asarray(result.pitch_hz,   dtype=np.float64)
    times  = np.asarray(result.timestamps, dtype=np.float64)
    voiced = np.asarray(result.voicing,    dtype=bool)

    f0_hz[~voiced] = 0.0

    pitch = np.column_stack([times, f0_hz])
    np.save(out_path, pitch)

    n_voiced = voiced.sum()
    print(
        f"[swiftf0] {len(pitch):,} frames  "
        f"({times[-1]:.1f} s)  "
        f"voiced: {n_voiced:,} ({n_voiced/len(pitch)*100:.1f}%)"
    )
    print(f"[swiftf0] → {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="SwiftF0 pitch extraction on Carnatic vocal audio."
    )
    parser.add_argument(
        "recordings", nargs="*",
        help="Recording ID(s). Defaults to CURRENT_PIECE; use --all for SARASUDA_VARNAM.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Process all recordings in settings.SARASUDA_VARNAM.",
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

    if args.all:
        recordings = settings.SARASUDA_VARNAM
    elif args.recordings:
        recordings = args.recordings
    else:
        recordings = [settings.CURRENT_PIECE]

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
            audio_path = _find_corpus_audio(recording_id)
            suffix = "original"

        print(f"\n── {recording_id}  [{suffix}] ──")
        predict_pitch(recording_id, audio_path, suffix)


if __name__ == "__main__":
    main()
