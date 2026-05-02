"""
Inference script for the Spectrogram-Channels U-Net (Oh et al. 2018).

Separates voice and tanpura from recordings in data/corpus/.

Usage:
    # Single recording:
    python -m src.source_separation.separate --recordings srs_v1_rkm_sav

    # Multiple recordings:
    python -m src.source_separation.separate --recordings srs_v1_rkm_sav srs_v1_svd_sav

    # All recordings defined in settings.py:
    python -m src.source_separation.separate --all

    # Custom checkpoint:
    python -m src.source_separation.separate --all --checkpoint data/interim/source_separation/checkpoint_best.pt

Outputs:
    data/interim/source_separation/separated/{recording_id}/{recording_id}_unet_voice.wav
    data/interim/source_separation/separated/{recording_id}/{recording_id}_unet_tanpura.wav
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import settings
from src.source_separation.unet import UNetSmall

# Must match training hyperparameters
SR           = 22050
N_FFT        = 1024
HOP_LENGTH   = 256
PATCH_FRAMES = 128
BASE         = 32

DEFAULT_CHECKPOINT = settings.DATA_INTERIM / "source_separation" / "checkpoint_best.pt"
OUT_ROOT           = settings.DATA_INTERIM / "source_separation" / "separated"


def _find_audio(recording_id: str) -> Path:
    audio_dir = settings.DATA_CORPUS / recording_id / "audio"
    for ext in ("wav", "mp3", "flac", "ogg"):
        candidates = list(audio_dir.glob(f"*.{ext}"))
        if candidates:
            return candidates[0]
    raise FileNotFoundError(f"No audio file found in {audio_dir}")


def _load_model(checkpoint_path: Path, device: torch.device) -> UNetSmall:
    model = UNetSmall(in_channels=1, out_channels=2, base=BASE).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def separate(
    audio_path: Path,
    model: UNetSmall,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Separate voice and tanpura from a single audio file."""
    audio, _ = librosa.load(audio_path, sr=SR, mono=True)

    D_mix   = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mag_mix = np.abs(D_mix)
    phase   = np.angle(D_mix)
    F, T    = mag_mix.shape

    n_patches = int(np.ceil(T / PATCH_FRAMES))
    T_padded  = n_patches * PATCH_FRAMES
    mag_pad   = np.pad(mag_mix, ((0, 0), (0, T_padded - T)))

    mag_voice   = np.zeros_like(mag_pad)
    mag_tanpura = np.zeros_like(mag_pad)

    with torch.no_grad():
        for i in range(n_patches):
            start = i * PATCH_FRAMES
            end   = start + PATCH_FRAMES

            patch = mag_pad[:, start:end]
            norm  = patch.max() + 1e-8
            x     = torch.tensor((patch / norm)[np.newaxis, np.newaxis], dtype=torch.float32).to(device)
            pred  = model(x)[0].cpu().numpy()   # (2, F, PATCH_FRAMES)

            mag_voice[:, start:end]   = pred[0] * norm
            mag_tanpura[:, start:end] = pred[1] * norm

    mag_voice   = mag_voice[:, :T]
    mag_tanpura = mag_tanpura[:, :T]

    voice   = librosa.istft(mag_voice   * np.exp(1j * phase), hop_length=HOP_LENGTH, length=len(audio))
    tanpura = librosa.istft(mag_tanpura * np.exp(1j * phase), hop_length=HOP_LENGTH, length=len(audio))

    return voice, tanpura


def process_recording(recording_id: str, model: UNetSmall, device: torch.device):
    audio_path = _find_audio(recording_id)
    print(f"  {recording_id}  ← {audio_path.name}")

    voice, tanpura = separate(audio_path, model, device)

    out_dir = OUT_ROOT / recording_id
    out_dir.mkdir(parents=True, exist_ok=True)

    voice_path   = out_dir / f"{recording_id}_unet_voice.wav"
    tanpura_path = out_dir / f"{recording_id}_unet_tanpura.wav"

    sf.write(voice_path,   voice,   SR)
    sf.write(tanpura_path, tanpura, SR)

    print(f"    → {voice_path}")
    print(f"    → {tanpura_path}")


def main():
    parser = argparse.ArgumentParser(description="Separate voice and tanpura with the trained U-Net.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--recordings", nargs="+", metavar="ID",
                       help="One or more recording IDs (e.g. srs_v1_rkm_sav)")
    group.add_argument("--all", action="store_true",
                       help="Process all recordings defined in settings.SARASUDA_VARNAM")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:     {device}")
    print(f"Checkpoint: {args.checkpoint}")

    model = _load_model(args.checkpoint, device)

    recordings = settings.SARASUDA_VARNAM if args.all else args.recordings
    print(f"Processing {len(recordings)} recording(s)...\n")

    for rec_id in recordings:
        process_recording(rec_id, model, device)


if __name__ == "__main__":
    main()
