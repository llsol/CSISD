"""
BS-RoFormer voice separation via audio-separator.

Usage:
    python -m src.source_separation.separate_as --recordings srs_v1_svd_sav
    python -m src.source_separation.separate_as --all
    python -m src.source_separation.separate_as --all --model model_bs_roformer_ep_368_sdr_12.9628.ckpt

Outputs:
    data/interim/source_separation_as/separated/{id}/{id}_as_voice.wav
    data/interim/source_separation_as/separated/{id}/{id}_as_instrumental.wav
"""

from __future__ import annotations

import argparse
import sys
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import settings

DEFAULT_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
OUT_ROOT      = settings.DATA_INTERIM / "source_separation_as" / "separated"


def _find_audio(recording_id: str) -> Path:
    audio_dir = settings.DATA_CORPUS / recording_id / "audio"
    for ext in ("wav", "mp3", "flac", "ogg"):
        candidates = list(audio_dir.glob(f"*.{ext}"))
        if candidates:
            return candidates[0]
    raise FileNotFoundError(f"No audio file found in {audio_dir}")


def process_recording(recording_id: str, model_filename: str):
    from audio_separator.separator import Separator

    audio_path = _find_audio(recording_id)
    out_dir    = OUT_ROOT / recording_id
    out_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = out_dir / "_tmp"
    tmp_dir.mkdir(exist_ok=True)

    print(f"  {recording_id}  ← {audio_path.name}")

    sep = Separator(output_dir=str(tmp_dir), output_format="wav")
    sep.load_model(model_filename)

    # audio-separator names outputs as {stem}_(Vocals)_{model}.wav etc.
    output_files = sep.separate(str(audio_path))

    # Rename to our convention
    voice_path        = out_dir / f"{recording_id}_as_voice.wav"
    instrumental_path = out_dir / f"{recording_id}_as_instrumental.wav"

    for f in output_files:
        src = Path(f)
        if not src.is_absolute():
            src = tmp_dir / src.name
        name_lower = src.name.lower()
        if "vocal" in name_lower:
            shutil.move(str(src), voice_path)
            print(f"    → {voice_path}")
        else:
            shutil.move(str(src), instrumental_path)
            print(f"    → {instrumental_path}")

    # Clean up tmp dir if empty
    try:
        tmp_dir.rmdir()
    except OSError:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Separate voice with BS-RoFormer (audio-separator)."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--recordings", nargs="+", metavar="ID")
    group.add_argument("--all", action="store_true",
                       help="Process all recordings in settings.SARASUDA_VARNAM")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"BS-RoFormer model filename (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    recordings = settings.SARASUDA_VARNAM if args.all else args.recordings
    print(f"Model:      {args.model}")
    print(f"Processing {len(recordings)} recording(s)...\n")

    for rec_id in recordings:
        process_recording(rec_id, args.model)


if __name__ == "__main__":
    main()
