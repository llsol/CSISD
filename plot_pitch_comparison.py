"""
Compara l'extracció de pitch original (corpus) amb la nova (interim, FTA-Net sobre àudio net).

Ús: python plot_pitch_comparison.py [recording_id] [t_start] [t_end]
    python plot_pitch_comparison.py srs_v1_svd_sav 10 40
"""

import sys
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from pathlib import Path

import settings

RECORDING_ID = sys.argv[1] if len(sys.argv) > 1 else settings.CURRENT_PIECE
T_START      = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
T_END        = float(sys.argv[3]) if len(sys.argv) > 3 else 60.0

# --- paths ---
corpus_pitch = (
    settings.DATA_CORPUS / RECORDING_ID / "raw" / f"{RECORDING_ID}_pitch_ftanet.tsv"
)
interim_pitch = (
    settings.DATA_INTERIM / RECORDING_ID / "pitch_raw" / f"{RECORDING_ID}_reproduction_ftanet_raw.npy"
)

# --- load original (corpus) ---
orig = np.loadtxt(corpus_pitch)          # (N, 2): time, f0_Hz
t_orig, f_orig = orig[:, 0], orig[:, 1]

# --- load new (interim, from clean audio) ---
new = np.load(interim_pitch)             # (N, 2): time, f0_Hz
t_new, f_new = new[:, 0], new[:, 1]

# --- window ---
def window(t, f, t0, t1):
    mask = (t >= t0) & (t <= t1) & (f > 0)
    return t[mask], f[mask]

to, fo = window(t_orig, f_orig, T_START, T_END)
tn, fn = window(t_new,  f_new,  T_START, T_END)

# --- plot ---
fig, axes = plt.subplots(2, 1, figsize=(16, 6), sharex=True, sharey=True)
fig.suptitle(f"{RECORDING_ID}  |  {T_START:.1f}s – {T_END:.1f}s", fontsize=12)

axes[0].scatter(to, fo, s=0.5, c="steelblue", rasterized=True)
axes[0].set_ylabel("f0 (Hz)")
axes[0].set_title("Original (corpus, àudio brut)")

axes[1].scatter(tn, fn, s=0.5, c="tomato", rasterized=True)
axes[1].set_ylabel("f0 (Hz)")
axes[1].set_xlabel("temps (s)")
axes[1].set_title("Nou (àudio net de tampura)")

plt.tight_layout()
out = Path(f"plot_pitch_comparison_{RECORDING_ID}_{int(T_START)}-{int(T_END)}s.png")
plt.savefig(out, dpi=150)
print(f"Guardat: {out}")
plt.show()
