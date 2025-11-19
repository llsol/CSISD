import os
from pathlib import Path

PROJECT = Path(".")   # Nom del projecte nou

# 7 peces del dataset (del screenshot)
CORPUS_PIECES = [
    "kmk_v1_ss_bhv",
    "rkb_v1_ss_bhv",
    "srs_v1_bdn_sav",
    "srs_v1_drn_sav",
    "srs_v1_psn_sav",
    "srs_v1_rkm_sav",
    "srs_v1_svd_sav",
]

# Svaras del sistema
SVARAS = ["sa", "ri", "ga", "ma", "pa", "dha", "ni"]

def make_dirs():

    # ─────────────────────────────────────────────
    # 1. DIRECTORIS GLOBALS DEL PROJECTE
    # ─────────────────────────────────────────────
    BASE_DIRS = [
        "data/corpus",

        "data/interim/pitch_interp",
        "data/interim/pitch_smooth",
        "data/interim/pitch_cents",
        "data/interim/svara_segments",
        "data/interim/derivatives",
        "data/interim/stability",

        "data/processed/svara_features",
        "data/processed/svara_templates",
        "data/processed/svara_stability",

        "synthetic/svaraforms",
        "synthetic/sequences",
        "synthetic/datasets/UCR_format",
        "synthetic/datasets/parquet",
        "synthetic/metadata",

        "src/io",
        "src/preprocessing",
        "src/svara",
        "src/features",
        "src/generation",
        "src/pipeline",
        "src/visualization",

        "notebooks/00_exploration",
        "notebooks/01_validation",
        "notebooks/02_svara_invariance",
        "notebooks/03_templates",
        "notebooks/04_synthesis",
        "notebooks/05_tsc_export",
        "notebooks/06_results",

        "models",
        "tests",
    ]

    for d in BASE_DIRS:
        (PROJECT / d).mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────
    # 2. CREAR LES 7 PECES DINS data/corpus/
    # ─────────────────────────────────────────────
    for piece in CORPUS_PIECES:
        base = PROJECT / "data/corpus" / piece

        # Subcarpetes MUSICALS originals
        subs = [
            "raw",          # raw propi de la peça
            "audio",
            "pitch",
            "annotations",
            "segments/svr",
            "segments/syn",
        ]

        for s in subs:
            (base / s).mkdir(parents=True, exist_ok=True)

        # ─────────────────────────────────────────────
        # 3. CREAR carpetes de SVARA dins sfrm/
        # sense subcarpetes sf
        # ─────────────────────────────────────────────
        for svara in SVARAS:
            svara_dir = base / "segments/svr" / svara
            svara_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProject tree successfully created under: {PROJECT}\n")


if __name__ == "__main__":
    make_dirs()
