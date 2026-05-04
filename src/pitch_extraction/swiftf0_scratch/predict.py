"""
Pitch extraction with SwiftF0-scratch (trained from scratch on SCMS Carnatic).

Writes:
    data/interim/{id}/pitch_raw_swiftf0scratch/{id}_{source}_swiftf0scratch_raw.npy
    shape: (N, 2) — col 0: time_sec, col 1: f0_Hz (0.0 = unvoiced)

Usage:
    python -m src.pitch_extraction.swiftf0_scratch.predict srs_v1_svd_sav
    python -m src.pitch_extraction.swiftf0_scratch.predict srs_v1_svd_sav --unet
    python -m src.pitch_extraction.swiftf0_scratch.predict srs_v1_svd_sav --as
    python -m src.pitch_extraction.swiftf0_scratch.predict --all
    python -m src.pitch_extraction.swiftf0_scratch.predict srs_v1_svd_sav --run run_002
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import settings
from src.pitch_extraction.swiftf0_scratch.model import SwiftF0Scratch, HOP, SR

CKPT_DIR             = settings.DATA_INTERIM / "models" / "swiftf0_scratch"
CONFIDENCE_THRESHOLD = 0.9
CHUNK_SEC            = 30.0


def _find_best_checkpoint(run_dir: Path | None = None) -> Path:
    if run_dir is None:
        runs = sorted(CKPT_DIR.glob("run_*"))
        if not runs:
            raise FileNotFoundError(f"No training runs found in {CKPT_DIR}")
        run_dir = runs[-1]
    best = run_dir / "best.pt"
    if not best.exists():
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
    recording_id:  str,
    audio_path:    Path,
    source_suffix: str,
    checkpoint:    Path,
    device:        torch.device,
    model:         SwiftF0Scratch,
) -> Path:
    import librosa

    out_dir = settings.DATA_INTERIM / recording_id / "pitch_raw_swiftf0scratch"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{recording_id}_{source_suffix}_swiftf0scratch_raw.npy"

    print(f"[swiftf0scratch] loading {audio_path.name} → resample to {SR} Hz")
    audio, _ = librosa.load(str(audio_path), sr=SR, mono=True)
    print(f"[swiftf0scratch] duration: {len(audio)/SR:.1f} s")

    chunk_samples = int(CHUNK_SEC * SR)
    all_times, all_f0 = [], []
    time_offset = 0.0
    hop_sec     = HOP / SR

    model.eval()
    with torch.no_grad():
        for start in range(0, len(audio), chunk_samples):
            chunk  = audio[start : start + chunk_samples]
            tensor = torch.from_numpy(chunk).unsqueeze(0).to(device)

            pitch_hz, confidence, _ = model(tensor)
            pitch_hz   = pitch_hz[0].cpu().numpy()
            confidence = confidence[0].cpu().numpy()

            n_frames = len(pitch_hz)
            times    = time_offset + np.arange(n_frames) * hop_sec
            voiced   = confidence >= CONFIDENCE_THRESHOLD
            f0_out   = pitch_hz.copy()
            f0_out[~voiced] = 0.0

            all_times.append(times)
            all_f0.append(f0_out)
            time_offset += n_frames * hop_sec

    times_cat = np.concatenate(all_times)
    f0_cat    = np.concatenate(all_f0)
    pitch     = np.column_stack([times_cat, f0_cat]).astype(np.float64)
    np.save(out_path, pitch)

    n_voiced = (f0_cat > 0).sum()
    print(
        f"[swiftf0scratch] {len(pitch):,} frames  "
        f"({times_cat[-1]:.1f} s)  "
        f"voiced: {n_voiced:,} ({n_voiced/len(pitch)*100:.1f}%)  "
        f"conf_thr={CONFIDENCE_THRESHOLD}"
    )
    print(f"[swiftf0scratch] checkpoint: {checkpoint}")
    print(f"[swiftf0scratch] → {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Pitch extraction with SwiftF0-scratch (Carnatic, from scratch)."
    )
    parser.add_argument("recordings", nargs="*",
                        help="Recording ID(s). Defaults to CURRENT_PIECE.")
    parser.add_argument("--all", action="store_true",
                        help="Process all recordings in settings.SARASUDA_VARNAM.")
    parser.add_argument("--run", default=None,
                        help="Training run directory name (e.g. run_002).")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--unet", action="store_true",
                        help="Use U-Net separated voice.")
    source.add_argument("--as", dest="as_model", action="store_true",
                        help="Use BS-RoFormer separated voice.")
    args = parser.parse_args()

    if args.all:
        recordings = settings.SARASUDA_VARNAM
    elif args.recordings:
        recordings = args.recordings
    else:
        recordings = [settings.CURRENT_PIECE]

    run_dir    = (CKPT_DIR / args.run) if args.run else None
    checkpoint = _find_best_checkpoint(run_dir)
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[swiftf0scratch] device={device}  checkpoint={checkpoint}")

    model = SwiftF0Scratch().to(device)
    ckpt  = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"[swiftf0scratch] Loaded epoch={ckpt['epoch']}  "
          f"val_loss={ckpt.get('best_val', '?'):.4f}")

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
        predict_pitch(recording_id, audio_path, source_suffix,
                      checkpoint, device, model)


if __name__ == "__main__":
    main()
