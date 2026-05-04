"""
Compare two pitch extractors on the same audio source.

Usage:
    # FTA-Net vs SwiftF0 (U-Net separated voice, all SARASUDA_VARNAM):
    python -m src.pitch_extraction.compare_pitch --all

    # A single recording, original audio:
    python -m src.pitch_extraction.compare_pitch srs_v1_svd_sav --source original

    # BS-RoFormer separated voice:
    python -m src.pitch_extraction.compare_pitch srs_v1_svd_sav --source as

    # Custom extractor pair:
    python -m src.pitch_extraction.compare_pitch srs_v1_svd_sav --extractor-a ftanet --extractor-b swiftf0 --source unet

Reads:
    data/interim/{id}/pitch_raw/{id}_{source}_{extractor}_raw.npy

Shows:
    1. Both pitch curves overlaid (cents re tonic)
    2. Absolute difference (voiced-in-both frames only)
    3. Signed difference
    4. Histogram of exclusive voiced frames
"""


from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import settings

# ── parameters ────────────────────────────────────────────────────────────────
TOP_N_REGIONS  = 5
REGION_SEC     = 10.0
VOICED_MIN_HZ  = 50.0
TONIC_FALLBACK = 200.0

EXTRACTOR_STYLES = {
    "ftanet":  ("FTA-Net",  "steelblue"),
    "swiftf0": ("SwiftF0",  "tomato"),
}

SOURCE_LABELS = {
    "original": "original corpus",
    "unet":     "U-Net separated",
    "as":       "BS-RoFormer separated",
}

# Configuració de resolucions per model (inferible dels fitxers, però útil per debug)
MODEL_CONFIG = {
    "ftanet":  {"sample_rate": 8000,  "hop_size": 80,   "time_res_ms": 10},
    "swiftf0": {"sample_rate": 16000, "hop_size": 256,  "time_res_ms": 16},
}
# ──────────────────────────────────────────────────────────────────────────────


def hz_to_cents(f0: np.ndarray, tonic: float) -> np.ndarray:
    out = np.full_like(f0, np.nan, dtype=float)
    voiced = f0 > VOICED_MIN_HZ
    out[voiced] = 1200.0 * np.log2(f0[voiced] / tonic)
    return out


def align_pitch_tracks(
    time_a: np.ndarray,
    pitch_a: np.ndarray,
    time_b: np.ndarray,
    pitch_b: np.ndarray,
    label_a: str = "A",
    label_b: str = "B",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Aligns two pitch tracks to the same time grid (the finer one).
    
    Returns:
        target_time: time grid (same for both)
        pitch_a_aligned: pitch_a interpolated to target grid (or original)
        pitch_b_aligned: pitch_b interpolated to target grid (or original)
        info_str: description of what was done
    """
    # Calcular resolució real (mostreig mitjà)
    dt_a = np.median(np.diff(time_a)) if len(time_a) > 1 else 0.01
    dt_b = np.median(np.diff(time_b)) if len(time_b) > 1 else 0.016
    
    # Tolerància per considerar resolucions iguals (0.5 ms)
    if abs(dt_a - dt_b) < 0.0005:
        # Ja estan alineats, només cal truncar a la mateixa longitud
        n = min(len(time_a), len(time_b))
        return time_a[:n], pitch_a[:n], pitch_b[:n], "same resolution (truncated)"
    
    # Determinar quina és la graella més fina (menor dt)
    if dt_a <= dt_b:
        # time_a és més fina → interpolarem pitch_b a la graella de time_a
        target_time = time_a
        f_interp = interpolate.interp1d(
            time_b, pitch_b,
            kind='linear',
            bounds_error=False,
            fill_value=np.nan
        )
        pitch_b_aligned = f_interp(target_time)
        pitch_a_aligned = pitch_a
        info = f"aligned: {label_b} ({dt_b*1000:.1f}ms) → {label_a} ({dt_a*1000:.1f}ms) grid"
    else:
        # time_b és més fina → interpolarem pitch_a
        target_time = time_b
        f_interp = interpolate.interp1d(
            time_a, pitch_a,
            kind='linear',
            bounds_error=False,
            fill_value=np.nan
        )
        pitch_a_aligned = f_interp(target_time)
        pitch_b_aligned = pitch_b
        info = f"aligned: {label_a} ({dt_a*1000:.1f}ms) → {label_b} ({dt_b*1000:.1f}ms) grid"
    
    return target_time, pitch_a_aligned, pitch_b_aligned, info


def _top_divergence_regions(
    time: np.ndarray,
    diff: np.ndarray,
    n: int,
    window_sec: float,
) -> list[tuple[float, float, float]]:
    dt = float(np.median(np.diff(time))) if len(time) > 1 else 0.01
    win = max(1, int(window_sec / dt))
    scores = []
    i = 0
    while i + win <= len(diff):
        valid_window = diff[i:i+win]
        valid_window = valid_window[np.isfinite(valid_window)]
        if len(valid_window) > 0:
            score = np.mean(np.abs(valid_window))
            scores.append((time[i], time[min(i + win - 1, len(time) - 1)], score))
        i += win
    scores.sort(key=lambda x: -x[2])
    selected: list[tuple[float, float, float]] = []
    for s, e, sc in scores:
        if all(e <= ps or s >= pe for ps, pe, _ in selected):
            selected.append((s, e, sc))
        if len(selected) == n:
            break
    return selected


def compare(
    recording_id: str,
    source: str = "unet",
    extractor_a: str = "ftanet",
    extractor_b: str = "swiftf0",
):
    pitch_dir = settings.DATA_INTERIM / recording_id / "pitch_raw"
    path_a = pitch_dir / f"{recording_id}_{source}_{extractor_a}_raw.npy"
    path_b = pitch_dir / f"{recording_id}_{source}_{extractor_b}_raw.npy"

    label_a, color_a = EXTRACTOR_STYLES.get(extractor_a, (extractor_a, "steelblue"))
    label_b, color_b = EXTRACTOR_STYLES.get(extractor_b, (extractor_b, "tomato"))
    source_label     = SOURCE_LABELS.get(source, source)

    if not path_a.exists():
        raise FileNotFoundError(
            f"{label_a} pitch not found: {path_a}\n"
            f"Run: python -m src.source_separation.ftanet_predict --{source}"
            if extractor_a == "ftanet" else
            f"Run: python -m src.pitch_extraction.swiftf0_predict --{source}"
        )
    if not path_b.exists():
        raise FileNotFoundError(
            f"{label_b} pitch not found: {path_b}\n"
            f"Run: python -m src.pitch_extraction.swiftf0_predict --{source}"
        )

    data_a = np.load(path_a)
    data_b = np.load(path_b)

    # Extreure temps i freqüències
    time_a = data_a[:, 0]
    time_b = data_b[:, 0]
    f0_a = data_a[:, 1]
    f0_b = data_b[:, 1]

    tonic = settings.SARASUDA_TONICS.get(recording_id, TONIC_FALLBACK)
    cents_a_raw = hz_to_cents(f0_a, tonic)
    cents_b_raw = hz_to_cents(f0_b, tonic)

    # ── ALINEACIÓ DE RESOLUCIONS (punt clau!) ────────────────────────────────
    time, cents_a, cents_b, align_info = align_pitch_tracks(
        time_a, cents_a_raw,
        time_b, cents_b_raw,
        label_a, label_b
    )
    
    # Calcular mètriques només on ambdós són finits (voiced)
    voiced_a = np.isfinite(cents_a)
    voiced_b = np.isfinite(cents_b)
    both_voiced = voiced_a & voiced_b

    diff = np.where(both_voiced, cents_b - cents_a, np.nan)
    mae  = np.nanmean(np.abs(diff))
    bias = np.nanmean(diff)
    
    total_frames = len(time)
    pct_both   = both_voiced.sum() / total_frames * 100
    pct_a_only = ((voiced_a & ~voiced_b).sum()) / total_frames * 100
    pct_b_only = ((~voiced_a & voiced_b).sum()) / total_frames * 100

    # Resolucions reals (per informació)
    dt_a_real = np.median(np.diff(time_a)) * 1000
    dt_b_real = np.median(np.diff(time_b)) * 1000

    print(f"\n{'─'*60}")
    print(f"Recording   : {recording_id}   [source: {source}]")
    print(f"Tonic       : {tonic:.1f} Hz")
    print(f"Duration    : {time[-1]:.1f} s")
    print(f"Resolution  : {label_a} = {dt_a_real:.1f}ms, {label_b} = {dt_b_real:.1f}ms")
    print(f"Alignment   : {align_info}")
    print(f"Total frames after alignment: {total_frames:,}")
    print(f"Both voiced : {pct_both:.1f}%")
    print(f"{label_a} only : {pct_a_only:.1f}%")
    print(f"{label_b} only : {pct_b_only:.1f}%")
    print(f"MAE (cents) : {mae:.1f}")
    print(f"Bias (cents): {bias:+.1f}  ({label_b} − {label_a})")
    print(f"{'─'*60}\n")

    regions = _top_divergence_regions(time, diff, TOP_N_REGIONS, REGION_SEC)

    a_only_cents = cents_a[voiced_a & ~voiced_b]
    b_only_cents = cents_b[~voiced_a & voiced_b]

    # ── plots ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 11))
    fig.suptitle(
        f"Pitch comparison — {recording_id}  "
        f"[{label_a} ({dt_a_real:.0f}ms) vs {label_b} ({dt_b_real:.0f}ms)]  "
        f"[{source_label}]  (tonic={tonic:.1f} Hz)",
        fontsize=13,
    )

    gs      = fig.add_gridspec(3, 2, width_ratios=[4, 1], hspace=0.35, wspace=0.08)
    ax0     = fig.add_subplot(gs[0, 0])
    ax1     = fig.add_subplot(gs[1, 0], sharex=ax0)
    ax2     = fig.add_subplot(gs[2, 0], sharex=ax0)
    ax_hist = fig.add_subplot(gs[:, 1])
    axes    = [ax0, ax1, ax2]

    # 1. Pitch curves
    ax0.plot(time, cents_a, lw=0.6, color=color_a, alpha=0.8, label=f"{label_a} ({dt_a_real:.0f}ms)")
    ax0.plot(time, cents_b, lw=0.6, color=color_b, alpha=0.8, label=f"{label_b} ({dt_b_real:.0f}ms)")
    ax0.set_ylabel("Cents re tonic")
    ax0.legend(loc="upper right", fontsize=8)
    ax0.set_title(f"Pitch curves — {label_a} vs {label_b}  [{source_label}]")

    # 2. Absolute difference
    ax1.fill_between(time, 0, np.abs(diff), where=np.isfinite(diff),
                     color="orchid", alpha=0.7)
    ax1.axhline(mae, color="purple", lw=1, ls="--", label=f"MAE={mae:.1f} ¢")
    ax1.set_ylabel("|Δ| cents")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_title(f"|Δ| pitch — {label_a} vs {label_b}  (both-voiced frames)")

    # 3. Signed difference
    ax2.fill_between(time, 0, diff, where=np.isfinite(diff) & (diff > 0),
                     color=color_b, alpha=0.6, label=f"{label_b} higher")
    ax2.fill_between(time, 0, diff, where=np.isfinite(diff) & (diff < 0),
                     color=color_a, alpha=0.6, label=f"{label_b} lower")
    ax2.axhline(0,    color="black",  lw=0.8)
    ax2.axhline(bias, color="purple", lw=1, ls="--", label=f"bias={bias:+.1f} ¢")
    ax2.set_ylabel("Δ cents")
    ax2.set_xlabel("Time (s)")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.set_title(f"Signed difference ({label_b} − {label_a})  [{source_label}]")

    # Highlight top divergence regions
    if regions:
        region_colors = plt.cm.autumn(np.linspace(0.2, 0.8, len(regions)))
        for ax in axes:
            for (s, e, _), col in zip(regions, region_colors):
                ax.axvspan(s, e, color=col, alpha=0.18)
        for i, ((s, e, sc), col) in enumerate(zip(regions, region_colors)):
            mid = (s + e) / 2
            axes[1].annotate(
                f"#{i+1}\n{sc:.0f}¢",
                xy=(mid, sc), xytext=(mid, sc + mae * 0.5),
                fontsize=7, ha="center", color="purple",
            )

    # 4. Histogram of exclusive voiced frames
    bins = np.arange(-1200, 2401, 50)
    if len(a_only_cents):
        ax_hist.barh(
            bins[:-1], np.histogram(a_only_cents, bins=bins)[0],
            height=48, color=color_a, alpha=0.7,
            label=f"{label_a} only ({pct_a_only:.1f}%)",
        )
    if len(b_only_cents):
        ax_hist.barh(
            bins[:-1], -np.histogram(b_only_cents, bins=bins)[0],
            height=48, color=color_b, alpha=0.7,
            label=f"{label_b} only ({pct_b_only:.1f}%)",
        )
    ax_hist.axhline(0, color="black", lw=0.5)
    ax_hist.axvline(0, color="black", lw=0.8)
    ax_hist.set_xlabel(f"Frames  (← {label_b}   {label_a} →)")
    ax_hist.set_ylabel("Cents re tonic")
    ax_hist.legend(fontsize=8, loc="upper right")
    ax_hist.set_title(
        f"Exclusive voiced frames\n{label_a} vs {label_b}", fontsize=9
    )

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Compare two pitch extractors on the same audio source."
    )
    parser.add_argument("recordings", nargs="*",
                        help="Recording IDs (default: all SARASUDA_VARNAM)")
    parser.add_argument("--all", action="store_true",
                        help="Process all recordings in settings.SARASUDA_VARNAM")
    parser.add_argument("--source", default="unet",
                        help="Audio source: original | unet | as  (default: unet)")
    parser.add_argument("--extractor-a", default="ftanet",
                        help="First extractor suffix (default: ftanet)")
    parser.add_argument("--extractor-b", default="swiftf0",
                        help="Second extractor suffix (default: swiftf0)")
    args = parser.parse_args()

    if args.all:
        recordings = settings.SARASUDA_VARNAM
    elif args.recordings:
        recordings = args.recordings
    else:
        recordings = settings.SARASUDA_VARNAM

    for rec_id in recordings:
        compare(
            rec_id,
            source=args.source,
            extractor_a=args.extractor_a,
            extractor_b=args.extractor_b,
        )


if __name__ == "__main__":
    main()