"""
FTA-Net pitch extraction on SCMS Carnatic audio (original + separated sources).

Run in the `compiam` environment:
    /home/lluis/anaconda3/envs/compiam/bin/python -m src.source_separation.scms.ftanet
    /home/lluis/anaconda3/envs/compiam/bin/python -m src.source_separation.scms.ftanet --source as
    /home/lluis/anaconda3/envs/compiam/bin/python -m src.source_separation.scms.ftanet --split test --source original

Sources:
    original  ← SCMS audio as-is          (data/datasets/scms/audio/{stem}.wav)
    as        ← BS-RoFormer separated voice (data/interim/scms_separated/{stem}/{stem}_as_voice.wav)
    unet      ← U-Net separated voice      (data/interim/scms_separated/{stem}/{stem}_unet_voice.wav)

Outputs:
    data/interim/scms_pitch/{stem}_{source}_ftanet_raw.npy
    shape: (N, 2) — col 0: time_sec, col 1: f0_Hz (0.0 = unvoiced)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import librosa
import compiam

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import settings
from src.pitch_extraction.swiftf0_finetune.dataset import scms_official_split

SCMS_ROOT  = settings.PROJECT_ROOT / "data" / "datasets" / "scms"
SEP_ROOT   = settings.DATA_INTERIM / "scms_separated"
OUT_ROOT   = settings.DATA_INTERIM / "scms_pitch"
MODEL_NAME = "melody:ftanet-carnatic"
MODEL_SR   = 8000


def _audio_path(stem: str, source: str) -> Path:
    if source == "original":
        return SCMS_ROOT / "audio" / f"{stem}.wav"
    elif source == "as":
        return SEP_ROOT / stem / f"{stem}_as_voice.wav"
    elif source == "unet":
        return SEP_ROOT / stem / f"{stem}_unet_voice.wav"
    else:
        raise ValueError(f"Unknown source: {source}")


def predict_stem(stem: str, source: str, model) -> Path:
    audio_path = _audio_path(stem, source)
    if not audio_path.exists():
        print(f"  [skip] {stem}/{source} — audio not found")
        return None

    out_dir  = OUT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}_{source}_ftanet_raw.npy"

    if out_path.exists():
        print(f"  [skip] {stem}/{source} — already predicted")
        return out_path

    audio, _ = librosa.load(str(audio_path), sr=MODEL_SR, mono=True)
    pitch = model.predict(audio, input_sr=MODEL_SR)
    pitch = np.asarray(pitch, dtype=np.float64)
    np.save(out_path, pitch)
    print(f"  {stem}/{source}  →  {len(pitch)} frames  →  {out_path.name}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="FTA-Net prediction on SCMS audio."
    )
    parser.add_argument("--source", choices=["original", "as", "unet", "all"],
                        default="all")
    parser.add_argument("--split", choices=["train", "test", "all"], default="all")
    parser.add_argument("--stems", nargs="+", metavar="STEM")
    parser.add_argument("--scms-root", default=None)
    args = parser.parse_args()

    scms_root = Path(args.scms_root) if args.scms_root else SCMS_ROOT
    train_stems, test_stems = scms_official_split(scms_root)

    if args.stems:
        stems = args.stems
    elif args.split == "train":
        stems = train_stems
    elif args.split == "test":
        stems = test_stems
    else:
        stems = train_stems + test_stems

    sources = ["original", "as", "unet"] if args.source == "all" else [args.source]

    print(f"Loading {MODEL_NAME} ...")
    model = compiam.load_model(MODEL_NAME)

    print(f"Stems: {len(stems)}  sources: {sources}\n")
    for stem in stems:
        for source in sources:
            predict_stem(stem, source, model)


if __name__ == "__main__":
    main()
