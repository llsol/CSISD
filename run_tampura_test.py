"""
Quick test: tampura separation on srs_v1_svd_sav.
Writes two files to data/interim/srs_v1_svd_sav/audio_notampura/:
  - srs_v1_svd_sav_clean.wav   (tampura attenuated)
  - srs_v1_svd_sav_residual.wav (what was removed, for listening)
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

from settings import SARASUDA_TONICS
from src.source_separation import separate_tampura

RECORDING_ID = "srs_v1_svd_sav"
AUDIO_IN  = Path(f"data/corpus/{RECORDING_ID}/audio/{RECORDING_ID}.mp3")
OUT_DIR   = Path(f"data/interim/{RECORDING_ID}/audio_notampura")
AUDIO_OUT = OUT_DIR / f"{RECORDING_ID}_clean.wav"
RESIDUAL  = OUT_DIR / f"{RECORDING_ID}_residual.wav"

SR = 22050
tonic = SARASUDA_TONICS[RECORDING_ID]

print(f"Input : {AUDIO_IN}")
print(f"Tonic : {tonic:.2f} Hz")

# Tampura-only segments (seconds): [start, end]
REFERENCE_SEGMENTS = [
    (0.0,     1.602),    # intro
    (210.233, 213.019),  # 3:30.233 – 3:33.019
    (446.511, 448.119),  # 7:26.511 – end
]

# --- separation ---
y_clean = separate_tampura(
    AUDIO_IN,
    AUDIO_OUT,
    reference_segments_sec=REFERENCE_SEGMENTS,
    sr=SR,
    n_fft=4096,
    hop_length=512,
    attenuation_db=30.0,
    mask_gamma=0.4,   # < 1 comprimeix la màscara cap amunt: harmonics aguts febles reben més atenuació
)

# --- residual (original − clean) so you can hear what was removed ---
y_orig, _ = librosa.load(AUDIO_IN, sr=SR, mono=True)
residual = y_orig - y_clean
sf.write(RESIDUAL, residual, SR)

print(f"\nDone.")
print(f"  Clean    → {AUDIO_OUT}")
print(f"  Residual → {RESIDUAL}  (listen to check what was removed)")
