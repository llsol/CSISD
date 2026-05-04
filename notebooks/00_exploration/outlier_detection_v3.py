"""
Outlier detection exploration — dual-version consensus + LOF.

Phase 1 — Dual-version consensus:
    Compare raw FTA-Net vs U-Net FTA-Net pitch. Flag frames where both detect
    voice but disagree by more than DISCREPANCY_THRESH cents.

Phase 2 — Local Outlier Factor:
    Build a per-frame feature vector from the merged pitch signal and run LOF
    to surface anomalous frames without hand-tuned thresholds.

Usage:
    python notebooks/00_exploration/outlier_detection_v3.py
    python notebooks/00_exploration/outlier_detection_v3.py srs_v1_rkm_sav
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.neighbors import LocalOutlierFactor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import settings

# ── parameters ──────────────────────────────────────────────────────────────
VOICED_MIN_HZ       = 50.0    # f0 below this → unvoiced
DISCREPANCY_THRESH  = 100.0   # cents: both voiced but disagree → uncertain
AGREE_THRESH        = 50.0   # cents: both voiced and close → high-confidence

LOF_N_NEIGHBORS     = 30      # LOF k
LOF_CONTAMINATION   = 0.04    # expected outlier fraction
FEATURE_HALF_WIN    = 6       # frames on each side for local stats (total 13)
# ────────────────────────────────────────────────────────────────────────────


def hz_to_cents(f0: np.ndarray, tonic: float) -> np.ndarray:
    out = np.full_like(f0, np.nan, dtype=float)
    voiced = f0 > VOICED_MIN_HZ
    out[voiced] = 1200.0 * np.log2(f0[voiced] / tonic)
    return out


# ── Phase 1: dual-version consensus ─────────────────────────────────────────

def phase1_merge(
    f0_raw: np.ndarray,
    f0_unet: np.ndarray,
    tonic: float,
) -> dict[str, np.ndarray]:
    """
    Returns merged pitch and per-frame confidence labels.

    Labels:
        0 = both unvoiced
        1 = raw only (unet missed it — possibly tanpura masking removed real voice)
        2 = unet only (raw missed it — tanpura was masking)
        3 = both voiced, agree  (high confidence)
        4 = both voiced, uncertain (large discrepancy → possible harmonic confusion)
    """
    raw_voiced  = f0_raw  > VOICED_MIN_HZ
    unet_voiced = f0_unet > VOICED_MIN_HZ
    both_voiced = raw_voiced & unet_voiced

    cents_raw  = hz_to_cents(f0_raw,  tonic)
    cents_unet = hz_to_cents(f0_unet, tonic)

    discrepancy = np.where(both_voiced, np.abs(cents_unet - cents_raw), np.nan)

    label = np.zeros(len(f0_raw), dtype=int)
    label[raw_voiced  & ~unet_voiced]                              = 1  # raw only
    label[~raw_voiced &  unet_voiced]                              = 2  # unet only
    label[both_voiced & (discrepancy <= AGREE_THRESH)]             = 3  # agree
    label[both_voiced & (discrepancy >  AGREE_THRESH)]             = 4  # uncertain

    # Merged pitch: prefer unet where it adds coverage; discard uncertain frames
    cents_merged = cents_unet.copy()
    # Where only raw has voice, keep raw
    cents_merged[label == 1] = cents_raw[label == 1]
    # Uncertain frames → NaN (let interpolation handle later)
    cents_merged[label == 4] = np.nan
    # Both unvoiced → NaN
    cents_merged[label == 0] = np.nan

    return {
        "cents_raw":    cents_raw,
        "cents_unet":   cents_unet,
        "cents_merged": cents_merged,
        "discrepancy":  discrepancy,
        "label":        label,
    }


# ── Phase 2: local features + LOF ───────────────────────────────────────────

def build_features(time: np.ndarray, cents: np.ndarray) -> np.ndarray:
    """
    Build a (N, 5) feature matrix from the merged pitch signal.

    Features per frame t:
        0  cents_t                         (absolute pitch)
        1  Δcents_{t-1 → t}               (instantaneous velocity)
        2  Δcents_{t-w → t+w} / (2w)      (smoothed velocity over half-window)
        3  std( cents_{t-w : t+w+1} )      (local variability)
        4  cents_t − median(cents_ctx)     (excursion from local context)

    NaN propagation: frames with NaN cents get NaN features throughout.
    """
    N = len(cents)
    w = FEATURE_HALF_WIN

    vel1 = np.full(N, np.nan)
    vel1[1:] = cents[1:] - cents[:-1]

    vel_smooth = np.full(N, np.nan)
    for i in range(w, N - w):
        if np.isfinite(cents[i - w]) and np.isfinite(cents[i + w]):
            vel_smooth[i] = (cents[i + w] - cents[i - w]) / (2 * w)

    local_std = np.full(N, np.nan)
    local_med = np.full(N, np.nan)
    for i in range(w, N - w):
        chunk = cents[i - w : i + w + 1]
        finite = chunk[np.isfinite(chunk)]
        if len(finite) >= 3:
            local_std[i] = float(np.std(finite))
            local_med[i] = float(np.median(finite))

    excursion = cents - local_med

    feats = np.column_stack([cents, vel1, vel_smooth, local_std, excursion])
    return feats


def phase2_lof(
    time: np.ndarray,
    cents_merged: np.ndarray,
) -> dict[str, np.ndarray]:
    feats = build_features(time, cents_merged)

    # Only fit LOF on frames with complete features
    finite_mask = np.all(np.isfinite(feats), axis=1)
    finite_idx  = np.where(finite_mask)[0]

    lof_score   = np.full(len(time), np.nan)
    is_lof_outlier = np.zeros(len(time), dtype=bool)

    if len(finite_idx) < LOF_N_NEIGHBORS + 1:
        print("[LOF] Not enough finite frames — skipping.")
        return {"feats": feats, "lof_score": lof_score, "is_lof_outlier": is_lof_outlier}

    clf = LocalOutlierFactor(
        n_neighbors=LOF_N_NEIGHBORS,
        contamination=LOF_CONTAMINATION,
        novelty=False,
    )
    clf.fit(feats[finite_idx])
    # negative_outlier_factor_: more negative = more anomalous
    raw_scores = clf.negative_outlier_factor_

    # Flip sign so higher = more anomalous
    anomaly_score = -raw_scores
    lof_score[finite_idx]      = anomaly_score
    is_lof_outlier[finite_idx] = clf.fit_predict(feats[finite_idx]) == -1

    print(f"[LOF] Fitted on {len(finite_idx):,} frames. "
          f"Outliers: {is_lof_outlier.sum():,} ({is_lof_outlier.mean()*100:.1f}%)")

    return {"feats": feats, "lof_score": lof_score, "is_lof_outlier": is_lof_outlier}


# ── Plotting ─────────────────────────────────────────────────────────────────

def _label_colors(sep_label: str) -> dict:
    return {
        0: ("lightgrey",  "both unvoiced"),
        1: ("steelblue",  "raw only"),
        2: ("tomato",     f"{sep_label} only"),
        3: ("seagreen",   "agree"),
        4: ("darkorange", "uncertain (large discrepancy)"),
    }


def plot_phase1(time, p1, recording_id, sep_label: str = "U-Net"):
    label_colors = _label_colors(sep_label)

    fig, axes = plt.subplots(3, 1, figsize=(18, 10), sharex=True)
    fig.suptitle(f"Phase 1 — Dual-version consensus  [{recording_id}]", fontsize=13)

    # Row 0: both pitch curves + merged
    ax = axes[0]
    ax.plot(time, p1["cents_raw"],    lw=0.5, color="steelblue", alpha=0.7, label="raw FTA-Net")
    ax.plot(time, p1["cents_unet"],   lw=0.5, color="tomato",    alpha=0.7, label=f"{sep_label} FTA-Net")
    ax.plot(time, p1["cents_merged"], lw=0.8, color="seagreen",  alpha=0.9, label="merged")
    ax.set_ylabel("Cents re tonic")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Pitch curves")

    # Row 1: discrepancy where both voiced
    ax = axes[1]
    disc = p1["discrepancy"]
    ax.fill_between(time, 0, disc,
                    where=np.isfinite(disc),
                    color="darkorange", alpha=0.6)
    ax.axhline(DISCREPANCY_THRESH, color="red",   lw=1, ls="--", label=f"uncertain >{DISCREPANCY_THRESH}¢")
    ax.axhline(AGREE_THRESH,       color="green", lw=1, ls="--", label=f"agree <{AGREE_THRESH}¢")
    ax.set_ylabel("|Δ| cents")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"Discrepancy between raw and {sep_label} (both-voiced frames)")

    # Row 2: label scatter
    ax = axes[2]
    label = p1["label"]
    for lv, (color, desc) in label_colors.items():
        mask = label == lv
        if not mask.any():
            continue
        ax.scatter(time[mask], np.full(mask.sum(), lv),
                   s=1, color=color, label=f"{lv}: {desc}", alpha=0.7)
    ax.set_yticks(list(label_colors))
    ax.set_yticklabels([d for _, d in label_colors.values()], fontsize=7)
    ax.set_xlabel("Time (s)")
    ax.set_title("Per-frame consensus label")
    ax.legend(loc="upper right", fontsize=7, markerscale=5)

    plt.tight_layout()
    plt.show()

    # ── Label distribution summary ──
    print("\nPhase 1 label distribution:")
    for lv, (_, desc) in label_colors.items():
        n = (label == lv).sum()
        print(f"  {lv} {desc:35s}: {n:7,}  ({n/len(label)*100:5.1f}%)")


def plot_phase2_features(time, feats_raw, feats_unet, recording_id, sep_label: str = "U-Net"):
    feat_names = [
        "cents (pitch)",
        "Δcents t-1→t (vel 1-frame)",
        "Δcents smooth (vel 6-frame)",
        "local std (±6 frames)",
        "excursion from local median",
    ]

    fig, axes = plt.subplots(len(feat_names), 1, figsize=(18, 11), sharex=True)
    fig.suptitle(f"Phase 2 — LOF input features  [{recording_id}]", fontsize=13)

    for i, (ax, name) in enumerate(zip(axes, feat_names)):
        ax.plot(time, feats_raw[:, i],  lw=0.55, color="#1a6fad", alpha=0.85, label="raw")
        ax.plot(time, feats_unet[:, i], lw=0.55, color="#d63a2a", alpha=0.75, label=sep_label)
        ax.set_ylabel(name, fontsize=8)
        ax.grid(axis="y", lw=0.3, alpha=0.35, color="grey")
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


def plot_phase2_lof(time, cents_merged, p2, recording_id, sep_label: str = "U-Net"):
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f"Phase 2 — LOF anomaly score  [{recording_id}]", fontsize=13)

    gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[3, 1], hspace=0.35, wspace=0.08)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    ax_hist = fig.add_subplot(gs[:, 1])   # histogram — independent x axis

    score  = p2["lof_score"]
    thresh = np.nanpercentile(score, (1 - LOF_CONTAMINATION) * 100)

    # Row 0: merged pitch coloured by LOF outlier
    normal_mask  = np.isfinite(cents_merged) & ~p2["is_lof_outlier"]
    outlier_mask = np.isfinite(cents_merged) &  p2["is_lof_outlier"]
    ax0.scatter(time[normal_mask],  cents_merged[normal_mask],
                s=0.5, color="seagreen", alpha=0.6, label="normal")
    ax0.scatter(time[outlier_mask], cents_merged[outlier_mask],
                s=4,   color="red",      alpha=0.9, label=f"LOF outlier ({outlier_mask.sum():,})")
    ax0.set_ylabel("Cents re tonic")
    ax0.legend(loc="upper right", fontsize=8)
    ax0.set_title(f"Merged pitch ({sep_label}) — LOF outliers highlighted")

    # Row 1: LOF score over time
    ax1.fill_between(time, 0, np.where(np.isfinite(score), score, 0),
                     color="orchid", alpha=0.7)
    ax1.axhline(thresh, color="red", lw=1, ls="--",
                label=f"threshold @ {LOF_CONTAMINATION*100:.0f}% contamination")
    ax1.set_ylabel("LOF score")
    ax1.set_xlabel("Time (s)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_title("LOF anomaly score over time")

    # Right: score histogram (independent x axis)
    finite_scores = score[np.isfinite(score)]
    ax_hist.hist(finite_scores, bins=120, color="orchid", alpha=0.8,
                 edgecolor="none", orientation="vertical")
    ax_hist.axvline(thresh, color="red",   lw=1.5, ls="--", label=f"threshold={thresh:.2f}")
    ax_hist.axvline(1.0,    color="black", lw=0.8, ls=":",  label="LOF=1")
    ax_hist.set_xlabel("LOF score")
    ax_hist.set_ylabel("Frame count")
    ax_hist.set_title("Score distribution", fontsize=9)
    ax_hist.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


# ── Main ─────────────────────────────────────────────────────────────────────

SEP_LABELS = {"unet": "U-Net", "as": "BS-RoFormer"}


def run(recording_id: str, sep: str = "unet"):
    sep_label = SEP_LABELS.get(sep, sep)
    pitch_dir = settings.DATA_INTERIM / recording_id / "pitch_raw"
    raw_path  = pitch_dir / f"{recording_id}_reproduction_ftanet_raw.npy"
    sep_path  = pitch_dir / f"{recording_id}_{sep}_ftanet_raw.npy"

    if not raw_path.exists():
        raise FileNotFoundError(f"Reproduction pitch not found: {raw_path}")
    if not sep_path.exists():
        raise FileNotFoundError(
            f"{sep_label} pitch not found: {sep_path}\n"
            f"Run: python -m src.source_separation.ftanet_predict --{sep}"
        )

    raw  = np.load(raw_path)  # (N, 2)
    sep_ = np.load(sep_path)  # (M, 2)

    n       = min(len(raw), len(sep_))
    time    = raw[:n, 0]
    f0_raw  = raw[:n, 1]
    f0_unet = sep_[:n, 1]

    tonic = settings.SARASUDA_TONICS.get(recording_id, 200.0)
    print(f"\nRecording : {recording_id}  [{sep_label}]")
    print(f"Tonic     : {tonic:.1f} Hz")
    print(f"Frames    : {n:,}  ({time[-1]:.1f} s)")

    title_id = f"{recording_id} [{sep_label}]"

    # ── Phase 1 ──
    print("\n── Phase 1: dual-version consensus ──")
    p1 = phase1_merge(f0_raw, f0_unet, tonic)
    plot_phase1(time, p1, title_id, sep_label=sep_label)

    # ── Phase 2 ──
    print("\n── Phase 2: LOF ──")
    p2 = phase2_lof(time, p1["cents_merged"])
    feats_raw  = build_features(time, p1["cents_raw"])
    feats_unet = build_features(time, p1["cents_unet"])
    plot_phase2_features(time, feats_raw, feats_unet, title_id, sep_label=sep_label)
    plot_phase2_lof(time, p1["cents_merged"], p2, title_id, sep_label=sep_label)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("recordings", nargs="*",
                        help="Recording IDs (default: all SARASUDA_VARNAM)")
    parser.add_argument("--as", dest="as_model", action="store_true",
                        help="Use BS-RoFormer separated pitch instead of U-Net")
    args = parser.parse_args()
    recordings = args.recordings or settings.SARASUDA_VARNAM
    sep = "as" if args.as_model else "unet"
    for recording_id in recordings:
        run(recording_id, sep=sep)


if __name__ == "__main__":
    main()
