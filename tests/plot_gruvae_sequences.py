"""
Visual smoke-test for src/models/gruvae/dataset.py

For a given recording, plots N svaras showing:
  - pitch curve (f0_savgol_p3_w13_cents)
  - segment shading (SIL=grey, CP=green, STA=orange)
  - representative pitch value per segment (dot + dashed line)

Run from project root:
    python tests/plot_gruvae_sequences.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import polars as pl

import settings as S
from src.io.pitch_io import load_preprocessed_pitch, load_flat_regions
from src.models.gruvae.dataset_gruvae import build_svara_sequences, INPUT_DIM

# ---------------------------------------------------------------------------
# Config — adjust as needed
# ---------------------------------------------------------------------------
RECORDING_ID = "srs_v1_svd_sav"
TONIC_HZ     = S.SARASUDA_TONICS[RECORDING_ID]
N_SVARAS     = 8       # how many svaras to plot
COLS         = 2       # subplot columns
FIGSIZE      = (14, 4)

SEG_COLORS = {"CP": "green", "SIL": "grey", "STA": "orange"}
SEG_ALPHA  = {"CP": 0.15,    "SIL": 0.18,  "STA": 0.12}

ONEHOT_TO_TYPE = {(1, 0, 0): "CP", (0, 1, 0): "SIL", (0, 0, 1): "STA"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def decode_sequence(seq: np.ndarray) -> list[dict]:
    """
    Decode a (n_segments, INPUT_DIM) array back to human-readable dicts.

    Columns:  [oh_CP, oh_SIL, oh_STA, dur_rel, total_dur_sec, cents_norm]
    """
    assert seq.shape[1] == INPUT_DIM
    segments = []
    for row in seq:
        oh = tuple(int(round(v)) for v in row[:3])
        seg_type   = ONEHOT_TO_TYPE.get(oh, "?")
        dur_rel    = float(row[3])
        total_dur  = float(row[4])
        cents      = float(row[5]) * 1200.0     # undo /1200 normalisation
        segments.append({
            "type":      seg_type,
            "dur_rel":   dur_rel,
            "total_dur": total_dur,
            "cents":     cents,
        })
    return segments


def plot_one_svara(ax, svara: dict, df_pitch: pl.DataFrame):
    """
    Plot one svara on ax:
      - pitch curve
      - segment colour bands
      - representative pitch dot per segment
    """
    t_start = svara["t_start"]
    t_end   = svara["t_end"]
    label   = svara["svara_label"]
    seq     = svara["sequence"]

    # Slice pitch to this svara
    t_all  = df_pitch["time_rel_sec"].to_numpy()
    p_all  = df_pitch["f0_savgol_p3_w13_cents"].to_numpy()
    mask   = (t_all >= t_start) & (t_all <= t_end)
    t_seg  = t_all[mask]
    p_seg  = p_all[mask]

    if len(t_seg) == 0:
        ax.set_title(f"{label} — no data")
        return

    # Pitch curve
    ax.plot(t_seg, p_seg, color="steelblue", lw=1.2, zorder=3)

    # Reconstruct absolute times using the actual total_dur stored in the sequence
    # (= times[-1] - times[0] of pitch samples, NOT t_end - t_start of annotation).
    # Also anchor t_cursor to t_seg[0], the first real pitch sample, not the annotation boundary.
    segments  = decode_sequence(seq)
    if not segments:
        return

    seq_total_dur = segments[0]["total_dur"]   # same value in every row
    t_cursor = t_seg[0]                         # first actual pitch sample

    p_finite = p_seg[np.isfinite(p_seg)]
    y_top = float(np.nanmax(p_finite)) + 30 if len(p_finite) else 0.0

    for seg in segments:
        seg_dur = seg["dur_rel"] * seq_total_dur
        t0  = t_cursor
        t1  = t_cursor + seg_dur
        t_mid = (t0 + t1) / 2.0

        # x position of the representative pitch dot:
        #   STA -> t0  (peak is always at the segment start)
        #   CP  -> t_mid (median of whole segment, no preferred x)
        #   SIL -> t_mid (inherited value, no preferred x)
        dot_x = t0 if seg["type"] == "STA" else t_mid

        # Colour band
        ax.axvspan(t0, t1,
                   color=SEG_COLORS.get(seg["type"], "purple"),
                   alpha=SEG_ALPHA.get(seg["type"], 0.1),
                   zorder=1)

        # Representative pitch dot + dashed horizontal line
        if np.isfinite(seg["cents"]):
            ax.plot(dot_x, seg["cents"],
                    marker="o", ms=5, zorder=5,
                    color=SEG_COLORS.get(seg["type"], "purple"))
            ax.hlines(seg["cents"], t0, t1,
                      colors=SEG_COLORS.get(seg["type"], "purple"),
                      lw=0.8, linestyles="--", alpha=0.6, zorder=4)

        # Type label at top
        ax.text(t_mid, y_top, seg["type"],
                ha="center", va="bottom",
                fontsize=6, color=SEG_COLORS.get(seg["type"], "purple"))

        t_cursor = t1

    ax.set_title(f"{label}  |  {len(segments)} segs  |  {seq_total_dur:.2f}s",
                 fontsize=9)
    ax.set_xlabel("time (s)", fontsize=7)
    ax.set_ylabel("cents", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.axhline(0, color="black", lw=0.5, ls="--", alpha=0.4)   # tonic


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Building sequences for {RECORDING_ID}...")
    svaras = build_svara_sequences(
        recording_id=RECORDING_ID,
        tonic_hz=TONIC_HZ,
    )
    print(f"  {len(svaras)} svaras found")

    # Load pitch for plotting
    df_pitch = load_preprocessed_pitch(
        recording_id=RECORDING_ID,
        tonic_hz=TONIC_HZ,
        convert_to_cents=True,
    )

    # Pick N_SVARAS evenly spaced across the recording
    step = max(1, len(svaras) // N_SVARAS)
    selected = svaras[::step][:N_SVARAS]

    rows = int(np.ceil(len(selected) / COLS))
    fig, axes = plt.subplots(rows, COLS, figsize=(FIGSIZE[0], FIGSIZE[1] * rows))
    axes = np.array(axes).flatten()

    for i, svara in enumerate(selected):
        plot_one_svara(axes[i], svara, df_pitch)

    # Hide unused subplots
    for j in range(len(selected), len(axes)):
        axes[j].set_visible(False)

    # Legend
    legend_handles = [
        mpatches.Patch(color=c, alpha=0.5, label=t)
        for t, c in SEG_COLORS.items()
    ]
    fig.legend(handles=legend_handles, loc="lower right", fontsize=8)

    fig.suptitle(
        f"GRU+VAE sequences — {RECORDING_ID}\n"
        f"dots = representative pitch per segment (cents/1200 decoded back to cents)",
        fontsize=10
    )
    plt.tight_layout()
    plt.show()

    # Print first svara decoded for quick inspection
    print("\n--- First svara decoded ---")
    s0 = selected[0]
    print(f"label={s0['svara_label']}  t=[{s0['t_start']:.2f}, {s0['t_end']:.2f}]")
    print(f"sequence shape: {s0['sequence'].shape}")
    for seg in decode_sequence(s0["sequence"]):
        print(f"  {seg['type']:3s}  dur_rel={seg['dur_rel']:.3f}  cents={seg['cents']:+.1f}")


if __name__ == "__main__":
    main()
