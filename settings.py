# settings.py
from pathlib import Path

CURRENT_PIECE = "srs_v1_bdn_sav"
PREPROCESSED_PITCH_PATH = Path("data/interim") / CURRENT_PIECE / "pitch" / f"{CURRENT_PIECE}_pitch_preprocessed.parquet"