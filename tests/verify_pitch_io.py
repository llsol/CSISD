"""
Quick smoke-test for src/io/pitch_io loaders.
Run from the project root:
    python tests/verify_pitch_io.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import settings as S

from src.io.pitch_io import (
    load_preprocessed_pitch,
    load_flat_regions,
    load_peaks,
)

RECORDING_ID = "srs_v1_bdn_sav"
TONIC_HZ = S.SARASUDA_TONICS[RECORDING_ID]


def check(label, fn):
    try:
        result = fn()
        print(f"  OK  {label}: {result.shape} | cols: {result.columns}")
    except FileNotFoundError:
        print(f"  --  {label}: fitxer no trobat (normal si no s'ha generat)")
    except Exception as e:
        print(f"  ERR {label}: {e}")


print(f"Recording: {RECORDING_ID}\n")

check("load_preprocessed_pitch (sense cents)",
      lambda: load_preprocessed_pitch(RECORDING_ID, convert_to_cents=False))

check("load_preprocessed_pitch (amb cents)",
      lambda: load_preprocessed_pitch(RECORDING_ID, tonic_hz=TONIC_HZ, convert_to_cents=True))

check("load_flat_regions",
      lambda: load_flat_regions(RECORDING_ID))

check("load_peaks",
      lambda: load_peaks(RECORDING_ID))
