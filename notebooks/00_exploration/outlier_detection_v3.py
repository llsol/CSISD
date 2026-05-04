"""
Outlier detection exploration — dual-version consensus + Hampel on increments.

Phase 1 — Dual-version consensus:
    Compare raw FTA-Net vs separated FTA-Net pitch. Flag frames where both
    detect voice but disagree by more than DISCREPANCY_THRESH cents.
    Note: label 4 ('uncertain') means incoherence between the two pitch curves,
    not necessarily a true outlier.

Phase 2 — Hampel on velocity (first differences):
    Apply the Hampel identifier to vel1 = Δcents/frame. A sudden harmonic jump
    creates two large opposite-sign vel1 spikes (entry + exit), both clearly
    flagged. HAMPEL_MIN_ABS_VEL guards against false positives in very stable
    (low-MAD) regions.

    score_i = |vel1_i − median(window_i)| / (1.4826 · MAD(window_i))
    outlier if score > HAMPEL_THRESHOLD  AND  |vel1_i| > HAMPEL_MIN_ABS_VEL

Usage:
    python notebooks/00_exploration/outlier_detection_v3.py
    python notebooks/00_exploration/outlier_detection_v3.py srs_v1_rkm_sav
    python notebooks/00_exploration/outlier_detection_v3.py srs_v1_svd_sav --as
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import settings

# ── parameters ──────────────────────────────────────────────────────────────
VOICED_MIN_HZ       = 50.0    # f0 below this → unvoiced
DISCREPANCY_THRESH  = 100.0   # cents: both voiced but disagree → uncertain
AGREE_THRESH        = 50.0    # cents: both voiced and close → high-confidence

FEATURE_HALF_WIN    = 6       # frames on each side for local stats (total 13)

HAMPEL_HALF_WIN     = 10      # frames on each side for Hampel window (total 21)
HAMPEL_THRESHOLD    = 3.0     # outlier if score > threshold (σ units)
HAMPEL_MIN_ABS_VEL  = 25.0   # ¢ — minimum |vel1| required to flag (prevents
                               #     false positives in very stable, low-MAD regions)
BIPOLAR_MAX_LAG     = 2       # max frames between entry and exit spike of a pair
# ────────────────────────────────────────────────────────────────────────────


def hz_to_cents(f0: np.ndarray, tonic: float) -> np.ndarray:
    out = np.full_like(f0, np.nan, dtype=float)
    voiced = f0 > VOICED_MIN_HZ
    out[voiced] = 1200.0 * np.log2(f0[voiced] / tonic)
    return out


# ── Phase 1: dual-version consensus ─────────────────────────────────────────

def phase1_merge(
    f0_raw: np.ndarray,
    f0_sep: np.ndarray,
    tonic: float,
) -> dict[str, np.ndarray]:
    """
    Returns merged pitch and per-frame confidence labels.

    Labels:
        0 = both unvoiced
        1 = raw only (separator missed it)
        2 = sep only (raw missed it — tanpura masking)
        3 = both voiced, agree  (high confidence)
        4 = both voiced, uncertain (large discrepancy → likely harmonic confusion,
                                    not necessarily an outlier)
    """
    raw_voiced  = f0_raw > VOICED_MIN_HZ
    sep_voiced  = f0_sep > VOICED_MIN_HZ
    both_voiced = raw_voiced & sep_voiced

    cents_raw = hz_to_cents(f0_raw, tonic)
    cents_sep = hz_to_cents(f0_sep, tonic)

    discrepancy = np.where(both_voiced, np.abs(cents_sep - cents_raw), np.nan)

    label = np.zeros(len(f0_raw), dtype=int)
    label[raw_voiced  & ~sep_voiced]                           = 1  # raw only
    label[~raw_voiced &  sep_voiced]                           = 2  # sep only
    label[both_voiced & (discrepancy <= AGREE_THRESH)]         = 3  # agree
    label[both_voiced & (discrepancy >  AGREE_THRESH)]         = 4  # uncertain

    cents_merged = cents_sep.copy()
    cents_merged[label == 1] = cents_raw[label == 1]
    cents_merged[label == 4] = np.nan
    cents_merged[label == 0] = np.nan

    return {
        "cents_raw":    cents_raw,
        "cents_sep":    cents_sep,
        "cents_merged": cents_merged,
        "discrepancy":  discrepancy,
        "label":        label,
    }


# ── Phase 2: feature computation ─────────────────────────────────────────────

def build_features(time: np.ndarray, cents: np.ndarray) -> np.ndarray:
    """
    Build a (N, 3) feature matrix from a pitch signal.

    Features per frame t:
        0  cents_t                         (absolute pitch)
        1  Δcents_{t-1 → t}               (instantaneous increment / vel1)
        2  cents_t − median(cents_ctx)     (excursion from local context)
    """
    N = len(cents)
    w = FEATURE_HALF_WIN

    vel1 = np.full(N, np.nan)
    vel1[1:] = cents[1:] - cents[:-1]

    local_med = np.full(N, np.nan)
    for i in range(w, N - w):
        chunk = cents[i - w : i + w + 1]
        finite = chunk[np.isfinite(chunk)]
        if len(finite) >= 3:
            local_med[i] = float(np.median(finite))

    excursion = cents - local_med

    return np.column_stack([cents, vel1, excursion])


# ── Phase 2: Hampel outlier detection ────────────────────────────────────────

def hampel_filter(
    signal: np.ndarray,
    half_win: int = HAMPEL_HALF_WIN,
    threshold: float = HAMPEL_THRESHOLD,
    min_abs: float = 0.0,
) -> dict[str, np.ndarray]:
    """
    Hampel identifier on any 1-D array (pitch, velocity, excursion, …).

    score_i = |signal_i − median(window_i)| / (1.4826 · MAD(window_i))

    Flags as outlier if:  score > threshold  AND  |signal_i| > min_abs
    The min_abs guard prevents false positives in very stable (low-MAD) regions
    where even a tiny fluctuation can score high.
    """
    N = len(signal)
    score        = np.full(N, np.nan)
    local_median = np.full(N, np.nan)
    is_outlier   = np.zeros(N, dtype=bool)

    for i in range(half_win, N - half_win):
        window = signal[i - half_win : i + half_win + 1]
        finite = window[np.isfinite(window)]
        if len(finite) < 3:
            continue
        med   = float(np.median(finite))
        mad   = float(np.median(np.abs(finite - med)))
        sigma = 1.4826 * mad
        local_median[i] = med
        if np.isfinite(signal[i]):
            s = abs(signal[i] - med) / (sigma + 1e-9)
            score[i]      = s
            is_outlier[i] = s > threshold and abs(signal[i]) > min_abs

    n_out   = int(is_outlier.sum())
    n_total = int(np.isfinite(signal).sum())
    print(
        f"[Hampel] window=±{half_win}fr  threshold={threshold}σ  "
        f"min_abs={min_abs}¢  "
        f"outliers: {n_out:,} / {n_total:,} ({n_out / max(n_total, 1) * 100:.1f}%)"
    )
    return {"score": score, "local_median": local_median, "is_outlier": is_outlier}


BIPOLAR_MAX_LAG = 2      # max distance (frames) between entry and exit spike


def bipolar_spike_detector(
    vel1: np.ndarray,
    min_abs: float = HAMPEL_MIN_ABS_VEL,
    max_lag: int = BIPOLAR_MAX_LAG,
) -> dict[str, np.ndarray]:
    """
    Detect harmonic-jump errors by finding pairs of opposite-sign vel1 spikes
    within max_lag frames of each other.

    Signature of a harmonic error:
        vel1[i]   >> 0  (entry: pitch jumps to wrong harmonic)
        vel1[i+k] << 0  (exit:  pitch returns), k ∈ {1, …, max_lag}

    pair_score[i] = min(|vel1[entry]|, |vel1[exit]|) — the smaller spike of the
    pair, which lower-bounds the size of the harmonic jump.

    Frames strictly between entry and exit are also flagged (they are the bad
    frames sitting at the wrong pitch).
    """
    N = len(vel1)
    is_outlier = np.zeros(N, dtype=bool)
    pair_score = np.zeros(N, dtype=float)

    large = (np.abs(vel1) > min_abs) & np.isfinite(vel1)

    for i in range(N):
        if not large[i]:
            continue
        sign_i = np.sign(vel1[i])
        for lag in range(1, max_lag + 1):
            j = i + lag
            if j >= N or not large[j]:
                continue
            if np.sign(vel1[j]) == -sign_i:
                score = float(min(abs(vel1[i]), abs(vel1[j])))
                for k in range(i, j + 1):
                    is_outlier[k] = True
                    pair_score[k] = max(pair_score[k], score)

    n_out = int(is_outlier.sum())
    print(
        f"[Bipolar] min_abs={min_abs}¢  max_lag={max_lag}fr  "
        f"outliers: {n_out:,} / {N:,} ({n_out / N * 100:.1f}%)"
    )
    return {"is_outlier": is_outlier, "pair_score": pair_score}


# ── Plotting ─────────────────────────────────────────────────────────────────

def _label_colors(sep_label: str) -> dict:
    return {
        0: ("lightgrey",  "both unvoiced"),
        1: ("steelblue",  "raw only"),
        2: ("tomato",     f"{sep_label} only"),
        3: ("seagreen",   "agree"),
        4: ("darkorange", "uncertain (incoherent between curves)"),
    }


def plot_phase1(time, p1, recording_id, sep_label: str = "U-Net"):
    label_colors = _label_colors(sep_label)

    fig, axes = plt.subplots(3, 1, figsize=(18, 10), sharex=True)
    fig.suptitle(f"Phase 1 — Dual-version consensus  [{recording_id}]", fontsize=13)

    ax = axes[0]
    ax.plot(time, p1["cents_raw"],    lw=0.5, color="steelblue", alpha=0.7, label="raw FTA-Net")
    ax.plot(time, p1["cents_sep"],    lw=0.5, color="tomato",    alpha=0.7, label=f"{sep_label} FTA-Net")
    ax.plot(time, p1["cents_merged"], lw=0.8, color="seagreen",  alpha=0.9, label="merged")
    ax.set_ylabel("Cents re tonic")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"Pitch curves — raw vs {sep_label}  [{recording_id}]")

    ax = axes[1]
    disc = p1["discrepancy"]
    ax.fill_between(time, 0, disc, where=np.isfinite(disc), color="darkorange", alpha=0.6)
    ax.axhline(DISCREPANCY_THRESH, color="red",   lw=1, ls="--", label=f"uncertain >{DISCREPANCY_THRESH}¢")
    ax.axhline(AGREE_THRESH,       color="green", lw=1, ls="--", label=f"agree <{AGREE_THRESH}¢")
    ax.set_ylabel("|Δ| cents")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"Discrepancy between raw and {sep_label} (both-voiced frames)")

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

    print("\nPhase 1 label distribution:")
    for lv, (_, desc) in label_colors.items():
        n = (label == lv).sum()
        print(f"  {lv} {desc:40s}: {n:7,}  ({n/len(label)*100:5.1f}%)")


def plot_phase2_features(time, feats_raw, feats_sep, recording_id, sep_label: str = "U-Net"):
    feat_names = [
        "cents (pitch)",
        "Δcents t-1→t (increment)",
        "excursion from local median",
    ]

    fig, axes = plt.subplots(len(feat_names), 1, figsize=(18, 9), sharex=True)
    fig.suptitle(f"Phase 2 — Pitch features  [{recording_id}]", fontsize=13)

    for i, (ax, name) in enumerate(zip(axes, feat_names)):
        ax.plot(time, feats_raw[:, i], lw=0.55, color="#1a6fad", alpha=0.85, label="raw")
        ax.plot(time, feats_sep[:, i], lw=0.55, color="#d63a2a", alpha=0.75, label=sep_label)
        ax.set_ylabel(name, fontsize=8)
        ax.grid(axis="y", lw=0.3, alpha=0.35, color="grey")
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


def plot_phase2_spikes(
    time, cents_merged, vel1_merged, p_hampel, p_bipolar, recording_id, sep_label: str = "U-Net"
):
    """
    Two-detector view:
      - Hampel (general large vel1 spikes, left column background)
      - Bipolar pairs (entry+exit within 1-2 frames, primary signal)
    """
    h_score      = p_hampel["score"]
    h_outlier    = p_hampel["is_outlier"]
    b_outlier    = p_bipolar["is_outlier"]
    pair_score   = p_bipolar["pair_score"]

    # Combined: either detector fires
    combined     = h_outlier | b_outlier

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        f"Phase 2 — Spike detection  [{recording_id}]  "
        f"(Hampel ±{HAMPEL_HALF_WIN}fr/{HAMPEL_THRESHOLD}σ  "
        f"+  Bipolar lag≤{BIPOLAR_MAX_LAG}fr  min|Δ|={HAMPEL_MIN_ABS_VEL}¢)",
        fontsize=12,
    )

    gs      = gridspec.GridSpec(3, 2, figure=fig, width_ratios=[3, 1], hspace=0.38, wspace=0.1)
    ax0     = fig.add_subplot(gs[0, 0])
    ax1     = fig.add_subplot(gs[1, 0], sharex=ax0)
    ax2     = fig.add_subplot(gs[2, 0], sharex=ax0)
    ax_hist = fig.add_subplot(gs[:, 1])

    # Row 0: pitch — normal / Hampel-only / bipolar pair
    normal_mask  = np.isfinite(cents_merged) & ~combined
    h_only_mask  = np.isfinite(cents_merged) & h_outlier & ~b_outlier
    bip_mask     = np.isfinite(cents_merged) & b_outlier

    ax0.scatter(time[normal_mask], cents_merged[normal_mask],
                s=0.5, color="seagreen", alpha=0.45, label="normal")
    if h_only_mask.any():
        ax0.scatter(time[h_only_mask], cents_merged[h_only_mask],
                    s=4, color="orchid", alpha=0.8, zorder=2,
                    label=f"Hampel only ({h_only_mask.sum():,})")
    if bip_mask.any():
        sc = ax0.scatter(
            time[bip_mask], cents_merged[bip_mask],
            c=pair_score[bip_mask], cmap="Reds",
            vmin=HAMPEL_MIN_ABS_VEL, vmax=HAMPEL_MIN_ABS_VEL * 5,
            s=8, alpha=0.9, zorder=3,
            label=f"bipolar pair ({bip_mask.sum():,})",
        )
        plt.colorbar(sc, ax=ax0, label="pair score (¢)", pad=0.01)
    ax0.set_ylabel("Cents re tonic")
    ax0.legend(loc="upper right", fontsize=8)
    ax0.set_title(f"Merged pitch ({sep_label}) — spike detectors")

    # Row 1: vel1 — normal / Hampel / bipolar
    normal_v = np.isfinite(vel1_merged) & ~combined
    h_only_v = np.isfinite(vel1_merged) & h_outlier & ~b_outlier
    bip_v    = np.isfinite(vel1_merged) & b_outlier

    ax1.plot(time[normal_v], vel1_merged[normal_v],
             ".", ms=0.8, color="steelblue", alpha=0.5, label="vel1 normal")
    if h_only_v.any():
        ax1.scatter(time[h_only_v], vel1_merged[h_only_v],
                    s=5, color="orchid", alpha=0.85, zorder=2,
                    label=f"Hampel only ({h_only_v.sum():,})")
    if bip_v.any():
        sc1 = ax1.scatter(
            time[bip_v], vel1_merged[bip_v],
            c=pair_score[bip_v], cmap="Reds",
            vmin=HAMPEL_MIN_ABS_VEL, vmax=HAMPEL_MIN_ABS_VEL * 5,
            s=10, alpha=0.9, zorder=3,
            label=f"bipolar ({bip_v.sum():,})",
        )
        plt.colorbar(sc1, ax=ax1, label="pair score (¢)", pad=0.01)
    ax1.axhline(0,                   color="black", lw=0.5)
    ax1.axhline( HAMPEL_MIN_ABS_VEL, color="grey",  lw=0.8, ls="--")
    ax1.axhline(-HAMPEL_MIN_ABS_VEL, color="grey",  lw=0.8, ls="--",
                label=f"±{HAMPEL_MIN_ABS_VEL}¢ threshold")
    ax1.set_ylabel("Δcents / frame  (vel1)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_title(
        f"Increments — bipolar pair = entry+exit spike within {BIPOLAR_MAX_LAG} frames"
    )

    # Row 2: pair_score over time (primary signal)
    ax2.fill_between(time, 0, pair_score, where=pair_score > 0,
                     color="tomato", alpha=0.65, label="bipolar pair score")
    finite_h = np.where(np.isfinite(h_score), h_score, 0)
    ax2.fill_between(time, 0, finite_h, where=(h_score > HAMPEL_THRESHOLD) & ~b_outlier,
                     color="orchid", alpha=0.5, label="Hampel score (non-bipolar)")
    ax2.axhline(HAMPEL_MIN_ABS_VEL, color="grey", lw=0.8, ls="--",
                label=f"min_abs={HAMPEL_MIN_ABS_VEL}¢")
    ax2.set_ylabel("Score (¢ / σ)")
    ax2.set_xlabel("Time (s)")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.set_title("Bipolar pair score (¢) + Hampel score (σ) over time")

    # Right: pair_score histogram
    nonzero_pair = pair_score[pair_score > 0]
    if len(nonzero_pair):
        ax_hist.hist(nonzero_pair, bins=60, color="tomato", alpha=0.75,
                     edgecolor="none", label="bipolar pairs")
    finite_h_scores = h_score[np.isfinite(h_score) & (h_score > 0)]
    if len(finite_h_scores):
        ax_hist.hist(finite_h_scores, bins=60, color="orchid", alpha=0.5,
                     edgecolor="none", label="Hampel scores")
    ax_hist.axvline(HAMPEL_MIN_ABS_VEL, color="grey", lw=1, ls="--",
                    label=f"{HAMPEL_MIN_ABS_VEL}¢")
    ax_hist.set_xlabel("Score")
    ax_hist.set_ylabel("Frame count")
    ax_hist.set_title("Score distributions", fontsize=9)
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

    raw  = np.load(raw_path)
    sep_ = np.load(sep_path)

    n      = min(len(raw), len(sep_))
    time   = raw[:n, 0]
    f0_raw = raw[:n, 1]
    f0_sep = sep_[:n, 1]

    tonic = settings.SARASUDA_TONICS.get(recording_id, 200.0)
    print(f"\nRecording : {recording_id}  [{sep_label}]")
    print(f"Tonic     : {tonic:.1f} Hz")
    print(f"Frames    : {n:,}  ({time[-1]:.1f} s)")

    title_id = f"{recording_id} [{sep_label}]"

    # ── Phase 1 ──
    print("\n── Phase 1: dual-version consensus ──")
    p1 = phase1_merge(f0_raw, f0_sep, tonic)
    plot_phase1(time, p1, title_id, sep_label=sep_label)

    # ── Phase 2 ──
    cents_merged = p1["cents_merged"]
    vel1_merged  = np.full(len(cents_merged), np.nan)
    vel1_merged[1:] = cents_merged[1:] - cents_merged[:-1]

    print("\n── Phase 2: spike detection ──")
    p_hampel  = hampel_filter(vel1_merged, min_abs=HAMPEL_MIN_ABS_VEL)
    p_bipolar = bipolar_spike_detector(vel1_merged)

    feats_raw = build_features(time, p1["cents_raw"])
    feats_sep = build_features(time, p1["cents_sep"])
    plot_phase2_features(time, feats_raw, feats_sep, title_id, sep_label=sep_label)
    plot_phase2_spikes(
        time, cents_merged, vel1_merged, p_hampel, p_bipolar, title_id, sep_label=sep_label
    )


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
