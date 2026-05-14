"""
BS-RoFormer voice separation via audio-separator.

CV mode (Carnatic Varnam corpus):
    python -m src.source_separation.separate_as srs_v1_svd_sav
    python -m src.source_separation.separate_as --all
    python -m src.source_separation.separate_as --all --model model_bs_roformer_ep_368_sdr_12.9628.ckpt

SCMS mode:
    python -m src.source_separation.separate_as --scms
    python -m src.source_separation.separate_as --scms --split test
    python -m src.source_separation.separate_as --scms --skip-existing

CV output  : data/interim/source_separation_as/separated/{id}/{id}_as_voice.wav
                                                                   {id}_as_instrumental.wav
SCMS output: data/interim/scms_separated/{stem}/{stem}_as_voice.wav
                                                 {stem}_as_instrumental.wav
"""

from __future__ import annotations

import argparse
import sys
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import settings
from src.pitch_extraction.swiftf0_finetune.dataset import scms_official_split

DEFAULT_MODEL   = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
CV_OUT_ROOT     = settings.INTERIM_SEP_CV_AS
SCMS_OUT_ROOT   = settings.INTERIM_SEP_SCMS
SCMS_AUDIO_DIR  = settings.PROJECT_ROOT / "data" / "datasets" / "scms" / "audio"
MIN_DURATION_S  = 3.0   # BS-RoFormer crashes on very short clips


def _find_cv_audio(recording_id: str) -> Path:
    audio_dir = settings.DATA_CORPUS / recording_id / "audio"
    for ext in ("wav", "mp3", "flac", "ogg"):
        candidates = list(audio_dir.glob(f"*.{ext}"))
        if candidates:
            return candidates[0]
    raise FileNotFoundError(f"No audio file found in {audio_dir}")


def _audio_duration(path: Path) -> float:
    import soundfile as sf
    info = sf.info(str(path))
    return info.duration


def separate(audio_path: Path, out_dir: Path, stem: str, model_filename: str) -> bool:
    """Returns True on success, False if skipped or failed."""
    from audio_separator.separator import Separator

    dur = _audio_duration(audio_path)
    if dur < MIN_DURATION_S:
        print(f"  {stem}  [skip — too short: {dur:.2f}s < {MIN_DURATION_S}s]")
        return False

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "_tmp"
    tmp_dir.mkdir(exist_ok=True)

    print(f"  {stem}  ← {audio_path.name}  ({dur:.1f}s)")

    try:
        sep = Separator(output_dir=str(tmp_dir), output_format="wav")
        sep.load_model(model_filename)
        output_files = sep.separate(str(audio_path))
    except Exception as e:
        print(f"  [ERROR] {stem}: {e}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return False

    voice_path        = out_dir / f"{stem}_as_voice.wav"
    instrumental_path = out_dir / f"{stem}_as_instrumental.wav"

    for f in output_files:
        src = Path(f)
        if not src.is_absolute():
            src = tmp_dir / src.name
        if "vocal" in src.name.lower():
            shutil.move(str(src), voice_path)
            print(f"    → {voice_path}")
        else:
            shutil.move(str(src), instrumental_path)
            print(f"    → {instrumental_path}")

    try:
        tmp_dir.rmdir()
    except OSError:
        pass
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Separate voice with BS-RoFormer (audio-separator)."
    )
    parser.add_argument("recordings", nargs="*", metavar="ID",
                        help="Recording ID(s) (CV mode). Defaults to CURRENT_PIECE.")
    parser.add_argument("--all", action="store_true",
                        help="Process all recordings in settings.SARASUDA_VARNAM (CV mode).")
    parser.add_argument("--scms", action="store_true",
                        help="Process SCMS clips instead of CV corpus recordings.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip clips whose voice output already exists.")
    parser.add_argument("--split", choices=["train", "test", "all"], default="all",
                        help="SCMS split to process (default: all).")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"BS-RoFormer model filename (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    print(f"Model: {args.model}\n")

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

        n_total = len(stems)
        n_done = n_skipped = n_failed = 0
        print(f"SCMS mode: {n_total} clips → {SCMS_OUT_ROOT}")

        for i, stem in enumerate(stems):
            out_dir    = SCMS_OUT_ROOT / stem
            voice_path = out_dir / f"{stem}_as_voice.wav"
            if args.skip_existing and voice_path.exists():
                n_skipped += 1
                continue
            audio_path = SCMS_AUDIO_DIR / f"{stem}.wav"
            print(f"\n── [{i+1}/{n_total}] {stem} ──")
            ok = separate(audio_path, out_dir, stem, args.model)
            if ok:
                n_done += 1
            else:
                n_failed += 1

        print(f"\nSCMS done: {n_done} ok, {n_skipped} skipped, {n_failed} failed/short.")
        return

    # ── CV mode ────────────────────────────────────────────────────────────────
    if args.all:
        recordings = settings.SARASUDA_VARNAM
    elif args.recordings:
        recordings = args.recordings
    else:
        recordings = [settings.CURRENT_PIECE]

    print(f"CV mode: {len(recordings)} recording(s) → {CV_OUT_ROOT}")
    for recording_id in recordings:
        audio_path = _find_cv_audio(recording_id)
        out_dir    = CV_OUT_ROOT / recording_id
        if args.skip_existing and (out_dir / f"{recording_id}_as_voice.wav").exists():
            print(f"  {recording_id}  [skip]")
            continue
        print(f"\n── {recording_id} ──")
        separate(audio_path, out_dir, recording_id, args.model)



if __name__ == "__main__":
    main()
