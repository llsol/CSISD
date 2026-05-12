"""
Pitch extraction with fine-tuned SwiftF0 (Carnatic SCMS).

Corpus mode — writes per recording:
    data/interim/cv_pitch_swiftf0finetune/{source}/{id}/{id}_{source}_swiftf0finetune-{thr}_raw.npy

SCMS mode (--scms) — writes per fragment:
    data/interim/scms_pitch_swiftf0finetune/{source}/{fragment}_{source}_swiftf0-finetune_raw.npy

Shape of every .npy: (N, 2) — col 0: time_sec, col 1: f0_Hz (0.0 = unvoiced)

Usage:
    python -m src.pitch_extraction.swiftf0_finetune.predict srs_v1_svd_sav
    python -m src.pitch_extraction.swiftf0_finetune.predict srs_v1_svd_sav --unet
    python -m src.pitch_extraction.swiftf0_finetune.predict srs_v1_svd_sav --as
    python -m src.pitch_extraction.swiftf0_finetune.predict --all
    python -m src.pitch_extraction.swiftf0_finetune.predict srs_v1_svd_sav --run run_002

    python -m src.pitch_extraction.swiftf0_finetune.predict --scms
    python -m src.pitch_extraction.swiftf0_finetune.predict --scms --split test
    python -m src.pitch_extraction.swiftf0_finetune.predict --scms --split test --thr 0.75
    python -m src.pitch_extraction.swiftf0_finetune.predict --scms --skip-existing
"""

from __future__ import annotations

import argparse
from html import parser
import sys
from pathlib import Path

import numpy as np
import torch

from src.pitch_extraction.swiftf0_finetune import model

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import settings
from src.pitch_extraction.swiftf0_finetune.model import SwiftF0, HOP, SR
from src.pitch_extraction.swiftf0_finetune.dataset import scms_official_split

CKPT_DIR             = settings.DATA_INTERIM / "models" / "swiftf0_carnatic"
CONFIDENCE_THRESHOLD = 0.9
SR                   = 16_000
CHUNK_SEC            = 30.0
SCMS_AUDIO_DIR       = settings.PROJECT_ROOT / "data" / "datasets" / "scms" / "audio"
SCMS_SEPARATED_DIR   = settings.DATA_INTERIM / "scms_separated"
SCMS_PITCH_DIR       = settings.DATA_INTERIM / "scms_pitch_swiftf0finetune"

def _find_best_checkpoint(run_dir: Path | None = None) -> Path:
    if run_dir is None:
        runs = sorted(CKPT_DIR.glob("run_*"))
        if not runs:
            raise FileNotFoundError(f"No training runs found in {CKPT_DIR}")
        run_dir = runs[-1]

    best = run_dir / "best.pt"
    if not best.exists():
        # Fall back to last.pt if best.pt doesn't exist yet
        best = run_dir / "last.pt"
    if not best.exists():
        raise FileNotFoundError(f"No checkpoint found in {run_dir}")
    return best


def _find_corpus_audio(recording_id: str) -> Path:
    audio_dir = settings.DATA_CORPUS / recording_id / "audio"
    for ext in ("mp3", "wav", "flac", "ogg"):
        candidates = list(audio_dir.glob(f"*.{ext}"))
        if candidates:
            return candidates[0]
    raise FileNotFoundError(f"No audio file found in {audio_dir}")


def predict_pitch(
    recording_id: str,
    audio_path: Path,
    source_suffix: str,
    checkpoint: Path,
    device: torch.device,
    model: SwiftF0,
    thr: float = CONFIDENCE_THRESHOLD,
    out_path: Path | None = None,
) -> Path:
    
    import librosa

    if out_path is None:
        out_dir = settings.DATA_INTERIM / "cv_pitch_swiftf0finetune" / source_suffix / recording_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{recording_id}_{source_suffix}_swiftf0finetune_raw.npy"

    print(f"[swiftf0ft] loading {audio_path.name} → resample to {SR} Hz")
    audio, _ = librosa.load(str(audio_path), sr=SR, mono=True)
    total_sec = len(audio) / SR
    print(f"[swiftf0ft] duration: {total_sec:.1f} s")

    chunk_samples = int(CHUNK_SEC * SR)
    all_times, all_f0 = [], []
    time_offset = 0.0

    model.eval()
    with torch.no_grad():
        for start in range(0, len(audio), chunk_samples):
            chunk = audio[start : start + chunk_samples]
            tensor = torch.from_numpy(chunk).unsqueeze(0).to(device)  # (1, L)

            pitch_hz, confidence, _ = model(tensor)
            # (1, T) → (T,)
            pitch_hz   = pitch_hz[0].cpu().numpy()
            confidence = confidence[0].cpu().numpy()

            n_frames  = len(pitch_hz)
            hop_sec = HOP / SR  # 0.016 s
            times     = time_offset + np.arange(n_frames) * hop_sec

            voiced         = confidence >= thr
            f0_out         = pitch_hz.copy()
            f0_out[~voiced] = 0.0

            all_times.append(times)
            all_f0.append(f0_out)
            time_offset += n_frames * hop_sec

    times_cat = np.concatenate(all_times)
    f0_cat    = np.concatenate(all_f0)
    pitch     = np.column_stack([times_cat, f0_cat]).astype(np.float64)
    np.save(out_path, pitch)

    voiced_mask = f0_cat > 0.0
    n_voiced    = voiced_mask.sum()
    print(
        f"[swiftf0ft] {len(pitch):,} frames  "
        f"({times_cat[-1]:.1f} s)  "
        f"voiced: {n_voiced:,} ({n_voiced/len(pitch)*100:.1f}%)  "
        f"conf_thr={CONFIDENCE_THRESHOLD}"
    )
    print(f"[swiftf0ft] checkpoint: {checkpoint}")
    print(f"[swiftf0ft] → {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Pitch extraction with fine-tuned SwiftF0 (SCMS Carnatic)."
    )
    parser.add_argument("recordings", nargs="*",
                        help="Recording ID(s). Defaults to CURRENT_PIECE.")
    parser.add_argument("--all", action="store_true",
                        help="Process all recordings in settings.SARASUDA_VARNAM.")
    parser.add_argument("--run", default=None,
                        help="Training run directory name (e.g. run_002). "
                             "Defaults to the latest run.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--unet", action="store_true",
                        help="Use U-Net separated voice.")
    source.add_argument("--as", dest="as_model", action="store_true",
                        help="Use BS-RoFormer separated voice.")
    parser.add_argument("--scms", action="store_true",
                    help="Process SCMS clips in data/datasets/scms/audio/.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip clips whose output .npy already exists.")
    parser.add_argument("--thr", type=float, default=CONFIDENCE_THRESHOLD,
                        help=f"Voicing confidence threshold (default: {CONFIDENCE_THRESHOLD})")
    parser.add_argument("--split", choices=["train", "test", "all"], default="all",
                        help="SCMS split to process.")
    args = parser.parse_args()

    thr = args.thr

    run_dir    = (CKPT_DIR / args.run) if args.run else None
    checkpoint = _find_best_checkpoint(run_dir)
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[swiftf0ft] device={device}  checkpoint={checkpoint}")

    model = SwiftF0().to(device)
    ckpt  = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"[swiftf0ft] Loaded epoch={ckpt['epoch']}  val_loss={ckpt.get('best_val', '?'):.4f}")


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
        n_total = len(stems)
        n_done  = 0

        print(f"[swiftf0finetune] SCMS mode: {n_total} clips  suffix={suffix} → {out_subdir}")

        for i, stem in enumerate(stems):
            out_path = out_subdir / f"{stem}_{suffix}_swiftf0-finetune_raw.npy"

            if args.skip_existing and out_path.exists():
                continue

            if args.as_model:
                audio_path = SCMS_SEPARATED_DIR / stem / f"{stem}_as_voice.wav"
            else:
                audio_path = SCMS_AUDIO_DIR / f"{stem}.wav"

            print(f"\n── [{i+1}/{n_total}] {stem}  thr={thr} ──")
            predict_pitch("_scms", audio_path, suffix, checkpoint, device, model,
                          thr=thr, out_path=out_path)
            n_done += 1

        print(f"\n[swiftf0finetune] SCMS done: {n_done} processed, {n_total - n_done} skipped.")
        return

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
            source_suffix = "unet"
        elif args.as_model:
            audio_path = (
                settings.DATA_INTERIM / "source_separation_as" / "separated"
                / recording_id / f"{recording_id}_as_voice.wav"
            )
            source_suffix = "as"
        else:
            audio_path    = _find_corpus_audio(recording_id)
            source_suffix = "original"

        print(f"\n── {recording_id}  [{source_suffix}] ──")
        predict_pitch(recording_id, audio_path, source_suffix, checkpoint, device, model)


if __name__ == "__main__":
    main()
