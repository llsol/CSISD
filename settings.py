# settings.py
from pathlib import Path


# Paths

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_CORPUS = PROJECT_ROOT / "data" / "corpus"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"



# Corpus / pieces

SARASUDA_VARNAM = [
    "srs_v1_bdn_sav",
    "srs_v1_drn_sav",
    "srs_v1_psn_sav",
    "srs_v1_rkm_sav",
    "srs_v1_svd_sav",
]

SARASUDA_TONICS = {
    "srs_v1_bdn_sav": 138.59,
    "srs_v1_drn_sav": 200.58,
    "srs_v1_psn_sav": 146.83,
    "srs_v1_rkm_sav": 149.40,
    "srs_v1_svd_sav": 210.07,
    "srs_v1_vgn_sav": 138.59,
}

CURRENT_PIECE = "srs_v1_bdn_sav"



# Column name conventions

TIME_COL = "time_rel_sec"

# pitch
PITCH_COL = "f0_savgol_p3_w13"       
PITCH_COL_CENTS = "f0_savgol_p3_w13_cents"
PITCH_COL_HZ = "pitch_hz"
PITCH_UNIT = "cents"                   # "cents" or "hz"

# ids / labels
RECORDING_ID_COL = "recording_id"
SECTION_ID_COL = "section_id"
SVARA_ID_COL = "svara_id"

SVARA_START_LABEL_COL = "svara_start_label"
SVARA_END_LABEL_COL = "svara_end_label"

# stability / flatness


CANDIDATE_COL = "flat_candidate"
STABLE_COL = "flat_region"
D1_THRESHOLD = 1000.0
MIN_STABLE_DURATION_SEC = 0.05   # o el que vulguis
TOLERANCE_CENTS = 30.0

# opcional (si vols guardar-ho al parquet per inspecció)
ABS_DERIV1_COL = "abs_deriv1_cents_per_sec"




# KDE / peak detection params

KDE_BW_METHOD = 0.1
KDE_NUM_POINTS = 8000
KDE_PEAK_DISTANCE = 50



# Plot defaults

FIGSIZE_WIDE = (16, 5)
FIGSIZE_NARROW = (14, 4)
DEFAULT_CONTEXT_N = 1   # svara neighborhood context for plots



# Common paths derived from CURRENT_PIECE

PREPROCESSED_PITCH_PATH = (
    DATA_INTERIM
    / CURRENT_PIECE
    / "pitch"
    / f"{CURRENT_PIECE}_pitch_preprocessed.parquet"
)


# Dictionary of dictionaries for different ragas

RAGAM_SVARAS_CENTS = {
    "saveri": {
        "svarasthanas": ["S", "R1", "G2", "M1", "P", "D1", "N2", "Ṡ"],
        "intervals_cents": [0, 100, 200, 200, 200, 100, 200, 200],  
    }
}