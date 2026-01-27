# settings.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_CORPUS = PROJECT_ROOT / "data" / "corpus"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

SARASUDA_VARNAM = ['srs_v1_bdn_sav',
                   'srs_v1_drn_sav',
                   'srs_v1_psn_sav',
                   'srs_v1_rkm_sav',
                   'srs_v1_svd_sav',
                    ]

SARASUDA_TONICS = {
    'srs_v1_bdn_sav': 138.59,
    'srs_v1_drn_sav': 200.58,
    'srs_v1_psn_sav': 146.83,
    'srs_v1_rkm_sav': 149.4,
    'srs_v1_svd_sav': 210.07,
    'srs_v1_vgn_sav': 138.59,
}

CURRENT_PIECE = "srs_v1_bdn_sav"
PREPROCESSED_PITCH_PATH = DATA_INTERIM / CURRENT_PIECE / "pitch" / f"{CURRENT_PIECE}_pitch_preprocessed.parquet"
