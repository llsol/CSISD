# settings.py
from pathlib import Path


# Paths

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_CORPUS    = PROJECT_ROOT / "data" / "corpus"
DATA_INTERIM   = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

FIGURES_DIR    = PROJECT_ROOT / "figures"

INTERIM_RECORDINGS  = DATA_INTERIM / "recordings"
INTERIM_PITCH_CV    = DATA_INTERIM / "pitch_predictions" / "cv"
INTERIM_PITCH_SCMS  = DATA_INTERIM / "pitch_predictions" / "scms"
INTERIM_SEP_CV_UNET = DATA_INTERIM / "source_separation" / "cv_unet"
INTERIM_SEP_CV_AS   = DATA_INTERIM / "source_separation" / "cv_as"
INTERIM_SEP_SCMS    = DATA_INTERIM / "source_separation" / "scms"
INTERIM_ANALYSIS    = DATA_INTERIM / "analysis"
INTERIM_MODELS      = DATA_INTERIM / "models"

GRUVAE_VERSION       = "gruvae_v4"
GRUVAE_DIR           = INTERIM_MODELS / GRUVAE_VERSION
GRUVAE_CV_DIR        = INTERIM_MODELS / "gruvae_cv"
CURVE_VAE_DIR        = INTERIM_MODELS / "curve_vae"
PARAM_GRU_DIR        = INTERIM_MODELS / "param_gru"
SYNTHESIS_DIR        = INTERIM_MODELS / "synthesis"
SWIFTF0_SCRATCH_DIR  = INTERIM_MODELS / "swiftf0_scratch"
SWIFTF0_CARNATIC_DIR = INTERIM_MODELS / "swiftf0_carnatic"
SEP_UNET_DIR         = INTERIM_MODELS / "source_separation_unet"

SSD_ROOT = Path("/media/lluis/Extreme SSD")
SCMS_ROOT = DATA_CORPUS / "scms" 




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
PITCH_COL_RAW = "f0_Hz"
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
MIN_STABLE_DURATION_SEC = 0.080  # 80ms — Viraraghavan et al. ISMIR 2017
TOLERANCE_CENTS = 30.0
MAX_SLOPE_CENTS_PER_SEC = 100.0  # 1 semitone/sec — Viraraghavan et al. ISMIR 2017
MAX_RESIDUAL_CENTS = 25.0        # RMS deviation from best-fit line; None = disabled

# opcional (si vols guardar-ho al parquet per inspecció)
ABS_DERIV1_COL = "abs_deriv1_cents_per_sec"

# CP generation post-processing (cp_vae.py)
# Savitzky-Golay smoothing applied to generated curves before rejection filter
CP_SAVGOL_WINDOW    = 11   # must be odd; applied over L_CANONICAL=64 samples
CP_SAVGOL_POLYORDER = 3

# Rejection sampling: generated curves must pass the same flatness criteria
# as the GT CP curves (same params as extract_flat_regions above).
# TOLERANCE_CENTS, MAX_SLOPE_CENTS_PER_SEC, MAX_RESIDUAL_CENTS are reused.




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
    INTERIM_RECORDINGS
    / CURRENT_PIECE
    / "pitch"
    / f"{CURRENT_PIECE}_pitch_preprocessed.parquet"
)


# Naming convention mappings (for converting external dataset filenames)

RAGA_CODE = {
    "saveri": "sav",
    "sri": "sri",
    "abhogi": "abh",
    "begada": "beg",
    "kalyani": "kal",
    "mohanam": "moh",
    "sahana": "sah",
}

PERFORMER_CODE = {
    "dharini": "drn",
    "ramakrishnamurthy": "rkm",
    "vignesh": "vgn",
    "prasanna": "psn",
    "sreevidya": "svd",
    "badrinarayanan": "bdn",
}

PIECE_CODE = {
    "saveri": "srs",
    "kalyani": "van",
    "sri": "saa",
    "sahana": "kar",
    "mohanam": "nin",
    "begada": "int",
    "abhogi": "evv",
}

DEFAULT_VERSION = "v1"


# Dictionary of dictionaries for different ragas

RAGAM_SVARAS_CENTS = {
    "saveri": {
        "svarasthanas": ["S", "R1", "G2", "M1", "P", "D1", "N2", "Ṡ"],
        "intervals_cents": [0, 100, 200, 200, 200, 100, 200, 200],  
    }
}


# List of feature columns to use for autoencoder training

AE_FEATURE_COLUMNS = [
    "duration_sec",
    "pitch_median",
    "pitch_std",
    "pitch_range",
    "slope_global",
    "d1_std",
    "d2_std",
]