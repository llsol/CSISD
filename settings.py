# settings.py
from pathlib import Path

SARASUDA_VARNAM = ['srs_v1_bdn_sav',
                   'srs_v1_drn_sav',
                   'srs_v1_psn_sav',
                   'srs_v1_rkm_sav',
                   'srs_v1_svd_sav',
                    ]


CURRENT_PIECE = "srs_v1_bdn_sav"
PREPROCESSED_PITCH_PATH = Path("data/interim") / CURRENT_PIECE / "pitch" / f"{CURRENT_PIECE}_pitch_preprocessed.parquet"
