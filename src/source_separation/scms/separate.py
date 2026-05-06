"""
BS-RoFormer voice separation on SCMS Carnatic audio.

Run in the `as` environment:
    conda run -n as python -m src.source_separation.scms.separate
    conda run -n as python -m src.source_separation.scms.separate --split test
    conda run -n as python -m src.source_separation.scms.separate --stems ek_murki_f1 ...

Outputs:
    data/interim/scms_separated/{stem}/{stem}_as_voice.wav
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import settings
from src.pitch_extraction.swiftf0_finetune.dataset import scms_official_split

SCMS_ROOT     = settings.PROJECT_ROOT / "data" / "datasets" / "scms"
OUT_ROOT      = settings.DATA_INTERIM / "scms_separated"
DEFAULT_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"


def separate_stem(stem: str, model_filename: str):
    from audio_separator.separator import Separator

    audio_path = SCMS_ROOT / "audio" / f"{stem}.wav"
    if not audio_path.exists():
        print(f"  [skip] {stem} — audio not found")
        return

    out_dir = OUT_ROOT / stem
    voice_path = out_dir / f"{stem}_as_voice.wav"
    if voice_path.exists():
        print(f"  [skip] {stem} — already separated")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "_tmp"
    tmp_dir.mkdir(exist_ok=True)

    print(f"  {stem}  ← {audio_path.name}")
    sep = Separator(output_dir=str(tmp_dir), output_format="wav")
    sep.load_model(model_filename)
    output_files = sep.separate(str(audio_path))

    instrumental_path = out_dir / f"{stem}_as_instrumental.wav"
    for f in output_files:
        src = Path(f)
        if not src.is_absolute():
            src = tmp_dir / src.name
        if "vocal" in src.name.lower():
            shutil.move(str(src), voice_path)
            print(f"    → {voice_path.name}")
        else:
            shutil.move(str(src), instrumental_path)

    try:
        tmp_dir.rmdir()
    except OSError:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Separate SCMS audio with BS-RoFormer."
    )
    parser.add_argument("--split", choices=["train", "test", "all"], default="all")
    parser.add_argument("--stems", nargs="+", metavar="STEM",
                        help="Override: process specific stems only")
    parser.add_argument("--model", default=DEFAULT_MODEL)
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

    print(f"Model : {args.model}")
    print(f"Stems : {len(stems)}\n")
    for stem in stems:
        separate_stem(stem, args.model)


if __name__ == "__main__":
    main()
