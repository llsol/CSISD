"""
Mutual information and invariance analysis for svara descriptors.

Usage:
    python -m src.analysis.svara_mi_analysis --tag v1_TR
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import settings as S
from src.analysis.svara_segment_analysis import load_all, COLORS


def _save(fig, name: str, out_dir: Path, dpi: int = 120) -> None:
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path.name}")


def run_mi_analysis(df, svara_labels, performers, out_dir: Path) -> None:
    """Run full MI + invariance + AUROC analysis and save plots."""
    n_s = len(svara_labels)
    n_p = len(performers)
    SVARA_COLORS = {sl: plt.cm.tab10(i / max(n_s, 1)) for i, sl in enumerate(svara_labels)}
    # ── cell 57 ───────────────────────────────────────────────────
    from sklearn.feature_selection import mutual_info_classif

    # ── Feature definitions ──────────────────────────────────────────────────────
    # [A] segment-level  [B] occurrence-level totals  [C] run-level  [D] pitch
    STRUCT_FEATS = [
        'svara_dur_sec',                                          # [B] total svara duration
        'n_cp',   'cp_total_dur_sec', 'cp_frac',                 # [A/B] CP segment count + totals
        'n_sta',  'sta_total_dur_sec', 'sta_frac',               # [A/B] STA aggregate
        'n_stap', 'stap_total_dur_sec', 'stap_frac',             # [A/B] STAp (peaks)
        'n_stat', 'stat_total_dur_sec', 'stat_frac',             # [A/B] STAt (troughs)
        'n_tr',   'tr_total_dur_sec',  'tr_frac',                # [A/B] TR aggregate
        'n_tra',  'tra_total_dur_sec', 'tra_frac',               # [A/B] TRa (ascending)
        'n_trd',  'trd_total_dur_sec', 'trd_frac',               # [A/B] TRd (descending)
        'sil_total_dur_sec', 'sil_frac',                         # [B] SIL total
        'n_cp_runs',   'cp_run_mean_dur',                        # [C] CP run structure
        'n_sta_runs',  'sta_run_mean_dur',                       # [C] STA aggregate runs
        'n_stap_runs', 'stap_run_mean_dur',                      # [C] STAp runs
        'n_stat_runs', 'stat_run_mean_dur',                      # [C] STAt runs
        'n_tr_runs',   'tr_run_mean_dur',                        # [C] TR aggregate runs
        'n_tra_runs',  'tra_run_mean_dur',                       # [C] TRa runs
        'n_trd_runs',  'trd_run_mean_dur',                       # [C] TRd runs
    ]
    PITCH_FEATS = [
        'cp_mean_cents',
        'sta_mean_cents',         'sta_std_cents',
        'stap_mean_cents',        'stap_std_cents',
        'stat_mean_cents',        'stat_std_cents',
        'sta_peaks_mean_cents',   'sta_peaks_std_cents',
        'sta_valleys_mean_cents', 'sta_valleys_std_cents',
    ]
    ALL_FEATS = STRUCT_FEATS + PITCH_FEATS

    FEAT_LABELS = {
        'svara_dur_sec':          'svara dur (s)',
        'n_cp':                   'n CP segs',
        'cp_total_dur_sec':       'CP total dur (s)',
        'cp_frac':                'CP frac',
        'n_sta':                  'n STA segs',
        'sta_total_dur_sec':      'STA total dur (s)',
        'sta_frac':               'STA frac',
        'n_stap':                 'n STAp segs',
        'stap_total_dur_sec':     'STAp total dur (s)',
        'stap_frac':              'STAp frac',
        'n_stat':                 'n STAt segs',
        'stat_total_dur_sec':     'STAt total dur (s)',
        'stat_frac':              'STAt frac',
        'n_tr':                   'n TR segs',
        'tr_total_dur_sec':       'TR total dur (s)',
        'tr_frac':                'TR frac',
        'n_tra':                  'n TRa segs',
        'tra_total_dur_sec':      'TRa total dur (s)',
        'tra_frac':               'TRa frac',
        'n_trd':                  'n TRd segs',
        'trd_total_dur_sec':      'TRd total dur (s)',
        'trd_frac':               'TRd frac',
        'sil_total_dur_sec':      'SIL total dur (s)',
        'sil_frac':               'SIL frac',
        'n_cp_runs':              'n CP runs',
        'cp_run_mean_dur':        'CP run mean dur',
        'n_sta_runs':             'n STA runs',
        'sta_run_mean_dur':       'STA run mean dur',
        'n_stap_runs':            'n STAp runs',
        'stap_run_mean_dur':      'STAp run mean dur',
        'n_stat_runs':            'n STAt runs',
        'stat_run_mean_dur':      'STAt run mean dur',
        'n_tr_runs':              'n TR runs',
        'tr_run_mean_dur':        'TR run mean dur',
        'n_tra_runs':             'n TRa runs',
        'tra_run_mean_dur':       'TRa run mean dur',
        'n_trd_runs':             'n TRd runs',
        'trd_run_mean_dur':       'TRd run mean dur',
        'cp_mean_cents':          'CP mean cents',
        'sta_mean_cents':         'STA mean cents',
        'sta_std_cents':          'STA std cents',
        'stap_mean_cents':        'STAp mean ¢',
        'stap_std_cents':         'STAp std ¢',
        'stat_mean_cents':        'STAt mean ¢',
        'stat_std_cents':         'STAt std ¢',
        'sta_peaks_mean_cents':   'STA peaks mean ¢',
        'sta_peaks_std_cents':    'STA peaks std ¢',
        'sta_valleys_mean_cents': 'STA valleys mean ¢',
        'sta_valleys_std_cents':  'STA valleys std ¢',
    }

    # ── Build feature matrix ──────────────────────────────────────────────────────
    df_pd = df.to_pandas()
    y_svara = df_pd['svara_label'].values

    X = df_pd[ALL_FEATS].values
    mi_all = mutual_info_classif(
        np.nan_to_num(X), y_svara, discrete_features=False, random_state=42
    )
    mi_all_dict    = dict(zip(ALL_FEATS, mi_all))
    mi_struct_dict = {f: mi_all_dict[f] for f in STRUCT_FEATS}
    mi_pitch_dict  = {f: mi_all_dict[f] for f in PITCH_FEATS}
    print('MI(descriptor, svara_label):')
    for f, v in sorted(mi_all_dict.items(), key=lambda x: -x[1]):
        print(f'  {f:35s}  {v:.4f}')

    # ── cell 58 ───────────────────────────────────────────────────
    # ── Plot: MI bar chart (two runs side by side) ───────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    COLOR_STRUCT = '#4878cf'
    COLOR_PITCH  = '#d65f5f'

    # — Left: ALL features, sorted by MI descending —
    ax = axes[0]
    feats_sorted = sorted(ALL_FEATS, key=lambda f: -mi_all_dict[f])
    vals  = [mi_all_dict[f] for f in feats_sorted]
    cols  = [COLOR_PITCH if f in PITCH_FEATS else COLOR_STRUCT for f in feats_sorted]
    names = [FEAT_LABELS[f] for f in feats_sorted]

    bars = ax.barh(range(len(feats_sorted)), vals, color=cols, alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(feats_sorted)))
    ax.set_yticklabels(names, fontsize=3)
    ax.invert_yaxis()
    ax.set_xlabel('Mutual Information (nats)', fontsize=11)
    ax.set_title('All features  (structural + pitch)\nMI vs. svara_label', fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    for bar, v in zip(bars, vals):
        ax.text(v + 0.003, bar.get_y() + bar.get_height() / 2,
                f'{v:.3f}', va='center', fontsize=3)

    legend_handles = [
        mpatches.Patch(color=COLOR_STRUCT, label='Structural', alpha=0.85),
        mpatches.Patch(color=COLOR_PITCH,  label='Pitch (cents)', alpha=0.85),
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc='lower right')

    # — Right: STRUCTURAL only, sorted by MI descending —
    ax = axes[1]
    feats_sorted_s = sorted(STRUCT_FEATS, key=lambda f: -mi_struct_dict[f])
    vals_s  = [mi_struct_dict[f] for f in feats_sorted_s]
    names_s = [FEAT_LABELS[f] for f in feats_sorted_s]

    bars_s = ax.barh(range(len(feats_sorted_s)), vals_s, color=COLOR_STRUCT, alpha=0.85,
                     edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(feats_sorted_s)))
    ax.set_yticklabels(names_s, fontsize=3)
    ax.invert_yaxis()
    ax.set_xlabel('Mutual Information (nats)', fontsize=11)
    ax.set_title('Structural features only\nMI vs. svara_label', fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    for bar, v in zip(bars_s, vals_s):
        ax.text(v + 0.003, bar.get_y() + bar.get_height() / 2,
                f'{v:.3f}', va='center', fontsize=3)

    fig.suptitle('Step 1 — Global Discriminative Power\nMI(descriptor, svara_label)  |  n=2760 svara occurrences', fontsize=13)
    plt.tight_layout()
    _save(fig, "20_mi_bar", out_dir)

    # ── cell 60 ───────────────────────────────────────────────────
    # ── MI per performer ─────────────────────────────────────────────────────────
    mi_per_perf = {}   # {perf: {feat: mi_val}}

    for perf in performers:
        sub   = df_pd[df_pd['performer'] == perf].copy()
        y_p   = sub['svara_label'].values
        X_p   = sub[ALL_FEATS].copy()
        for col in ALL_FEATS:
            X_p[col] = X_p[col].fillna(X_p[col].median())
        X_p = X_p.fillna(0.0)   # remaining NaN (all-NaN cols) → 0

        mi_p = mutual_info_classif(X_p.values, y_p, random_state=42)
        mi_per_perf[perf] = dict(zip(ALL_FEATS, mi_p))
        print(f'{perf}  (n={len(sub)}):  '
              + '  '.join(f'{FEAT_LABELS[f][:12]}={mi_per_perf[perf][f]:.3f}' for f in ALL_FEATS))

    # ── Consistency summary ──────────────────────────────────────────────────────
    import pandas as pd

    mi_df = pd.DataFrame(mi_per_perf).T   # shape (n_performers, n_feats)
    consistency = pd.DataFrame({
        'mean_MI': mi_df.mean(),
        'std_MI':  mi_df.std(),
        'cv_MI':   mi_df.std() / (mi_df.mean() + 1e-9),   # coefficient of variation
    }).sort_values('mean_MI', ascending=False)

    print('\nConsistency table (sorted by mean MI desc):')
    print(consistency.to_string(float_format='{:.4f}'.format))

    # ── cell 61 ───────────────────────────────────────────────────
    # ── Plot A: heatmap (features × performers) ──────────────────────────────────
    feats_by_mean = consistency.index.tolist()   # sorted by mean MI desc

    fig, axes = plt.subplots(1, 3, figsize=(18, 6),
                             gridspec_kw={'width_ratios': [2.2, 1.2, 1.2]})

    # ─ Left: heatmap MI per performer ─
    ax = axes[0]
    mat = np.array([[mi_per_perf[p][f] for p in performers] for f in feats_by_mean])
    norm_h = mcolors.Normalize(vmin=0, vmax=mat.max())
    im = ax.imshow(mat, aspect='auto', cmap='YlOrRd', norm=norm_h)
    ax.set_xticks(range(len(performers)))
    ax.set_xticklabels(performers, fontsize=11)
    ax.set_yticks(range(len(feats_by_mean)))
    ax.set_yticklabels([FEAT_LABELS[f] for f in feats_by_mean], fontsize=3)
    ax.set_title('MI(descriptor, svara_label)\nper performer', fontsize=12)
    plt.colorbar(im, ax=ax, shrink=0.8, label='MI (nats)')

    cmap_h = plt.cm.YlOrRd
    for i, f in enumerate(feats_by_mean):
        for j, p in enumerate(performers):
            v = mat[i, j]
            rgba = cmap_h(norm_h(v))
            lum  = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=3,
                    color='black' if lum > 0.45 else 'white')

    # ─ Middle: mean MI + per-performer dots ─
    ax = axes[1]
    for i, f in enumerate(feats_by_mean):
        per_vals = [mi_per_perf[p][f] for p in performers]
        ax.errorbar(consistency.loc[f, 'mean_MI'], i,
                    xerr=consistency.loc[f, 'std_MI'],
                    fmt='s', color='#4878cf', markersize=6,
                    capsize=3, capthick=1.5, elinewidth=1.5,
                    markeredgecolor='black', markeredgewidth=0.5, zorder=5)
        for pi, (p, v) in enumerate(zip(performers, per_vals)):
            ax.scatter(v, i, color=COLORS.get(p, 'gray'), s=30, zorder=6,
                       edgecolors='black', linewidths=0.4, alpha=0.85)

    ax.set_yticks(range(len(feats_by_mean)))
    ax.set_yticklabels([FEAT_LABELS[f] for f in feats_by_mean], fontsize=3)
    ax.invert_yaxis()
    ax.set_xlabel('MI (nats)', fontsize=10)
    ax.set_title('Mean ± std\n(dots = each performer)', fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    ax.axvline(0, color='black', lw=0.6, alpha=0.4)
    perf_legend = [mpatches.Patch(color=COLORS.get(p, 'gray'), label=p) for p in performers]
    ax.legend(handles=perf_legend, fontsize=8, loc='lower right')

    # ─ Right: coefficient of variation (CV = std/mean) — lower = more consistent ─
    ax = axes[2]
    cv_sorted = consistency['cv_MI'].values
    mean_sorted = consistency['mean_MI'].values
    bar_colors = ['#2ca02c' if v < 0.5 else '#ff7f0e' if v < 1.0 else '#d62728'
                  for v in cv_sorted]

    bars = ax.barh(range(len(feats_by_mean)), cv_sorted,
                   color=bar_colors, alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(feats_by_mean)))
    ax.set_yticklabels([FEAT_LABELS[f] for f in feats_by_mean], fontsize=3)
    ax.invert_yaxis()
    ax.set_xlabel('CV = std / mean  (lower = more consistent)', fontsize=10)
    ax.set_title('Consistency across performers\n(CV)', fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    ax.axvline(0.5, color='#ff7f0e', ls='--', lw=1, alpha=0.7, label='CV=0.5')
    ax.axvline(1.0, color='#d62728', ls='--', lw=1, alpha=0.7, label='CV=1.0')
    ax.legend(fontsize=8)
    for bar, v in zip(bars, cv_sorted):
        ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{v:.2f}', va='center', fontsize=3)

    fig.suptitle('Step 2 — Per-Performer Consistency of MI(descriptor, svara_label)', fontsize=13)
    plt.tight_layout()
    _save(fig, "21_mi_heatmap_performer", out_dir)

    # ── cell 63 ───────────────────────────────────────────────────
    # ── MI(descriptor, performer) ────────────────────────────────────────────────
    y_perf = df_pd['performer'].values

    X_all    = df_pd[ALL_FEATS].fillna(df_pd[ALL_FEATS].median()).fillna(0.0)
    X_struct = df_pd[STRUCT_FEATS].fillna(df_pd[STRUCT_FEATS].median()).fillna(0.0)

    mi_perf_all    = mutual_info_classif(X_all.values,    y_perf, random_state=42)
    mi_perf_struct = mutual_info_classif(X_struct.values, y_perf, random_state=42)

    mi_perf_all_dict    = dict(zip(ALL_FEATS,    mi_perf_all))
    mi_perf_struct_dict = dict(zip(STRUCT_FEATS, mi_perf_struct))

    print('MI(descriptor, performer) — all features, sorted:')
    for f, v in sorted(mi_perf_all_dict.items(), key=lambda x: -x[1]):
        svara_mi = mi_all_dict[f]
        ratio    = svara_mi / (v + 1e-9)
        print(f'  {FEAT_LABELS[f]:<30s}  performer={v:.4f}  svara={svara_mi:.4f}  ratio={ratio:.1f}x')

    # ── cell 64 ───────────────────────────────────────────────────
    # ── Plot: three panels ───────────────────────────────────────────────────────
    # Features sorted by MI(descriptor, performer) descending (most performer-dependent first)
    feats_by_perf = sorted(ALL_FEATS, key=lambda f: -mi_perf_all_dict[f])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6),
                             gridspec_kw={'width_ratios': [1.4, 1.4, 1.4]})

    bar_cols = [COLOR_PITCH if f in PITCH_FEATS else COLOR_STRUCT for f in feats_by_perf]
    names_p  = [FEAT_LABELS[f] for f in feats_by_perf]

    # ─ Left: MI(descriptor, performer) ─
    ax = axes[0]
    vals_p = [mi_perf_all_dict[f] for f in feats_by_perf]
    bars = ax.barh(range(len(feats_by_perf)), vals_p, color=bar_cols, alpha=0.85,
                   edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(feats_by_perf)))
    ax.set_yticklabels(names_p, fontsize=3)
    ax.invert_yaxis()
    ax.set_xlabel('MI (nats)', fontsize=11)
    ax.set_title('MI(descriptor, performer)\n↑ high = performer-dependent', fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    for bar, v in zip(bars, vals_p):
        ax.text(v + 0.002, bar.get_y() + bar.get_height() / 2,
                f'{v:.3f}', va='center', fontsize=3)
    ax.legend(handles=[
        mpatches.Patch(color=COLOR_STRUCT, label='Structural', alpha=0.85),
        mpatches.Patch(color=COLOR_PITCH,  label='Pitch (cents)', alpha=0.85),
    ], fontsize=9, loc='lower right')

    # ─ Middle: MI(descriptor, svara_label) on the same feature order ─
    ax = axes[1]
    vals_s = [mi_all_dict[f] for f in feats_by_perf]
    bars_s = ax.barh(range(len(feats_by_perf)), vals_s, color=bar_cols, alpha=0.85,
                     edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(feats_by_perf)))
    ax.set_yticklabels(names_p, fontsize=3)
    ax.invert_yaxis()
    ax.set_xlabel('MI (nats)', fontsize=11)
    ax.set_title('MI(descriptor, svara_label)\n(same feature order)', fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    for bar, v in zip(bars_s, vals_s):
        ax.text(v + 0.002, bar.get_y() + bar.get_height() / 2,
                f'{v:.3f}', va='center', fontsize=3)

    # ─ Right: scatter MI(svara) vs MI(performer) — invariance map ─
    ax = axes[2]
    for f in ALL_FEATS:
        x = mi_perf_all_dict[f]
        y = mi_all_dict[f]
        color = COLOR_PITCH if f in PITCH_FEATS else COLOR_STRUCT
        ax.scatter(x, y, color=color, s=60, zorder=5,
                   edgecolors='black', linewidths=0.6, alpha=0.9)
        ax.annotate(FEAT_LABELS[f], (x, y),
                    textcoords='offset points', xytext=(5, 3),
                    fontsize=7.5, alpha=0.9)

    # Diagonal: MI(svara) == MI(performer) — below is "more svara than performer"
    lim = max(max(mi_perf_all_dict.values()), max(mi_all_dict.values())) * 1.15
    ax.plot([0, lim], [0, lim], 'k--', lw=0.8, alpha=0.4, label='MI equal')
    ax.fill_between([0, lim], [0, 0], [0, lim], alpha=0.04, color='green')
    ax.fill_between([0, lim], [0, lim], [lim, lim], alpha=0.04, color='red')
    ax.text(lim * 0.65, lim * 0.85, 'svara > performer\n(invariant zone)',
            fontsize=8, color='green', alpha=0.7)
    ax.text(lim * 0.35, lim * 0.1, 'performer > svara\n(style zone)',
            fontsize=8, color='firebrick', alpha=0.7)

    ax.set_xlim(-0.01, lim)
    ax.set_ylim(-0.01, lim)
    ax.set_xlabel('MI(descriptor, performer)  →  performer dependence', fontsize=10)
    ax.set_ylabel('MI(descriptor, svara_label)  →  discriminative power', fontsize=10)
    ax.set_title('Invariance map\n(want: top-left quadrant)', fontsize=12)
    ax.grid(alpha=0.2)
    ax.legend(handles=[
        mpatches.Patch(color=COLOR_STRUCT, label='Structural', alpha=0.85),
        mpatches.Patch(color=COLOR_PITCH,  label='Pitch (cents)', alpha=0.85),
    ], fontsize=9)

    fig.suptitle('Step 3 — Performer Dependence  |  MI(descriptor, performer)', fontsize=13)
    plt.tight_layout()
    _save(fig, "22_mi_performer_panels", out_dir)

    # ── cell 66 ───────────────────────────────────────────────────
    eps = 1e-9

    max_svara = max(mi_all_dict.values())
    max_perf  = max(mi_perf_all_dict.values())

    disc_norm = {f: mi_all_dict[f]      / max_svara for f in ALL_FEATS}
    perf_norm = {f: mi_perf_all_dict[f] / max_perf  for f in ALL_FEATS}
    inv_norm  = {f: 1.0 - perf_norm[f]              for f in ALL_FEATS}
    inv_score = {f: disc_norm[f] * inv_norm[f]       for f in ALL_FEATS}

    # ── Classification (data-driven thresholds: median disc_norm, median perf_norm) ──
    disc_vals = list(disc_norm.values())
    perf_vals = list(perf_norm.values())
    disc_thr = float(np.median(disc_vals))   # features above median discriminability
    perf_thr = float(np.median(perf_vals))   # features below median performer dependence

    def classify(f):
        d = disc_norm[f]
        p = perf_norm[f]
        if d >= disc_thr and p < perf_thr:
            return 'Invariant & discriminative'
        elif d >= disc_thr and p >= perf_thr:
            return 'Discriminative but performer-dependent'
        else:
            return 'Weak / non-informative'

    import pandas as pd
    summary = pd.DataFrame({
        'MI_svara':     {f: mi_all_dict[f]      for f in ALL_FEATS},
        'MI_performer': {f: mi_perf_all_dict[f] for f in ALL_FEATS},
        'disc_norm':    disc_norm,
        'perf_norm':    perf_norm,
        'inv_score':    inv_score,
        'category':     {f: classify(f) for f in ALL_FEATS},
    }).sort_values('inv_score', ascending=False)
    summary.index = [FEAT_LABELS[f] for f in summary.index]

    print(f'Thresholds: disc_norm ≥ {disc_thr:.3f}, perf_norm < {perf_thr:.3f}\n')
    print(summary[['MI_svara','MI_performer','disc_norm','perf_norm','inv_score','category']]
          .to_string(float_format='{:.4f}'.format))

    # ── cell 67 ───────────────────────────────────────────────────
    CAT_COLORS = {
        'Invariant & discriminative':           '#2ca02c',
        'Discriminative but performer-dependent':'#ff7f0e',
        'Weak / non-informative':               '#aec7e8',
    }

    feats_ranked = sorted(ALL_FEATS, key=lambda f: -inv_score[f])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6),
                             gridspec_kw={'width_ratios': [1.5, 1.3]})

    # ─ Left: inv_score bar chart ─
    ax = axes[0]
    vals_is  = [inv_score[f] for f in feats_ranked]
    bar_cats = [classify(f) for f in feats_ranked]
    bar_c    = [CAT_COLORS[c] for c in bar_cats]
    names_r  = [FEAT_LABELS[f] for f in feats_ranked]

    bars = ax.barh(range(len(feats_ranked)), vals_is, color=bar_c, alpha=0.88,
                   edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(feats_ranked)))
    ax.set_yticklabels(names_r, fontsize=3)
    ax.invert_yaxis()
    ax.set_xlabel('Invariance score  =  disc_norm × (1 − perf_norm)', fontsize=11)
    ax.set_title('Step 4 — Invariance Score\n(ranked, color = category)', fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    for bar, v in zip(bars, vals_is):
        ax.text(v + 0.003, bar.get_y() + bar.get_height() / 2,
                f'{v:.3f}', va='center', fontsize=3)

    cat_legend = [mpatches.Patch(color=c, label=k, alpha=0.88)
                  for k, c in CAT_COLORS.items()]
    ax.legend(handles=cat_legend, fontsize=8, loc='lower right')

    # ─ Right: 2D invariance map, bubble = inv_score ─
    ax = axes[1]
    for f in ALL_FEATS:
        x     = mi_perf_all_dict[f]
        y     = mi_all_dict[f]
        score = inv_score[f]
        cat   = classify(f)
        color = CAT_COLORS[cat]
        size  = 80 + 800 * score   # bubble proportional to score
        ax.scatter(x, y, s=size, color=color, alpha=0.82,
                   edgecolors='black', linewidths=0.7, zorder=5)
        ax.annotate(FEAT_LABELS[f], (x, y),
                    textcoords='offset points', xytext=(6, 3), fontsize=7.5)

    lim = max(max(mi_perf_all_dict.values()), max(mi_all_dict.values())) * 1.15
    ax.plot([0, lim], [0, lim], 'k--', lw=0.8, alpha=0.35)
    ax.set_xlim(-0.01, lim)
    ax.set_ylim(-0.01, lim)
    ax.axvline(max_perf * perf_thr,  ls=':', color='gray', lw=1, alpha=0.6,
               label=f'perf_norm={perf_thr:.2f}')
    ax.axhline(max_svara * disc_thr, ls=':', color='gray', lw=1, alpha=0.6,
               label=f'disc_norm={disc_thr:.2f}')
    ax.set_xlabel('MI(descriptor, performer)  →  performer dependence', fontsize=10)
    ax.set_ylabel('MI(descriptor, svara_label)  →  discriminative power', fontsize=10)
    ax.set_title('Invariance map  (bubble size ∝ inv_score)', fontsize=12)
    ax.grid(alpha=0.2)
    ax.legend(handles=cat_legend + ax.get_legend_handles_labels()[0][-2:],
              fontsize=8, loc='upper left')

    plt.tight_layout()
    _save(fig, "23_invariance_scatter", out_dir)

    # ── cell 69 ───────────────────────────────────────────────────
    # Top-ranked structural features + top pitch feature
    top_struct = sorted(STRUCT_FEATS, key=lambda f: -inv_score[f])[:4]
    top_pitch  = sorted(PITCH_FEATS,  key=lambda f: -inv_score[f])[:2]
    top_feats  = top_struct + top_pitch   # 6 panels

    ncols = 3
    nrows = int(np.ceil(len(top_feats) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5 * nrows), squeeze=False)

    for idx, f in enumerate(top_feats):
        ax = axes[idx // ncols][idx % ncols]
        cat = classify(f)

        # ── Violin per svara (all performers combined) ──
        data_per_svara = []
        for sl in svara_labels:
            vals = df_pd[df_pd['svara_label'] == sl][f].dropna().values
            data_per_svara.append(vals if len(vals) > 1 else np.array([np.nan]))

        parts = ax.violinplot(
            [v[np.isfinite(v)] for v in data_per_svara],
            positions=range(n_s),
            showmedians=True, showextrema=False, widths=0.7,
        )
        for pc, sl in zip(parts['bodies'], svara_labels):
            pc.set_facecolor(SVARA_COLORS[sl])
            pc.set_alpha(0.55)
        parts['cmedians'].set_color('black')
        parts['cmedians'].set_linewidth(1.5)

        # ── Per-performer medians (dots) ──
        for xi, sl in enumerate(svara_labels):
            for perf in performers:
                sub = df_pd[(df_pd['svara_label'] == sl) &
                            (df_pd['performer'] == perf)][f].dropna()
                if len(sub) == 0:
                    continue
                ax.scatter(xi, sub.median(), color=COLORS.get(perf, 'gray'),
                           s=35, zorder=6, edgecolors='black', linewidths=0.5, alpha=0.9)

        ax.set_xticks(range(n_s))
        ax.set_xticklabels(svara_labels, fontsize=13, fontweight='bold')
        ax.set_ylabel(FEAT_LABELS[f], fontsize=10)
        ax.set_title(f'{FEAT_LABELS[f]}\n[{cat}]',
                     fontsize=10, color=CAT_COLORS[cat], fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        score_txt = f'inv_score={inv_score[f]:.3f}'
        ax.text(0.98, 0.97, score_txt, transform=ax.transAxes,
                fontsize=8, ha='right', va='top', color='gray')

    # Hide empty axes
    for idx in range(len(top_feats), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    # Shared legend
    perf_legend = [mpatches.Patch(color=COLORS.get(p, 'gray'), label=p) for p in performers]
    fig.legend(handles=perf_legend, title='Performer (dot = median)',
               loc='lower right', fontsize=9, ncol=len(performers))

    fig.suptitle('Step 5 — Distribution per svara for top-ranked features\n'
                 'Violin = all performers combined  |  Dot = per-performer median',
                 fontsize=13)
    plt.tight_layout()
    _save(fig, "24_validation_scatter", out_dir)

    # ── cell 70 ───────────────────────────────────────────────────
    # ── Final classification table ───────────────────────────────────────────────
    print('=' * 72)
    print(f'{"FINAL DESCRIPTOR CLASSIFICATION":^72}')
    print('=' * 72)
    print(f'  Thresholds: disc_norm ≥ {disc_thr:.3f} | perf_norm < {perf_thr:.3f}')
    print()

    for cat in CAT_COLORS:
        feats_in_cat = [f for f in ALL_FEATS if classify(f) == cat]
        feats_in_cat_s = sorted(feats_in_cat, key=lambda f: -inv_score[f])
        print(f'  ▸ {cat}')
        for f in feats_in_cat_s:
            pitch_tag = '  [pitch]' if f in PITCH_FEATS else ''
            print(f'      {FEAT_LABELS[f]:<30s}  score={inv_score[f]:.3f}'
                  f'  MI_svara={mi_all_dict[f]:.4f}'
                  f'  MI_perf={mi_perf_all_dict[f]:.4f}{pitch_tag}')
        print()

    print('─' * 72)
    print()
    print('KEY CLAIMS:')
    print()
    print('  1. Structure alone (without pitch) carries discriminative information.')
    top_s = sorted(STRUCT_FEATS, key=lambda f: -mi_struct_dict[f])
    for f in top_s[:3]:
        print(f'     → {FEAT_LABELS[f]}: MI_svara(struct)={mi_struct_dict[f]:.4f}')
    print()
    print('  2. Descriptors classified as "Invariant & discriminative":')
    inv_disc = [f for f in ALL_FEATS if classify(f) == 'Invariant & discriminative']
    for f in sorted(inv_disc, key=lambda f: -inv_score[f]):
        print(f'     → {FEAT_LABELS[f]}: score={inv_score[f]:.3f}')
    print()
    print('  3. Descriptors influenced by performance style:')
    perf_dep = [f for f in ALL_FEATS if classify(f) == 'Discriminative but performer-dependent']
    for f in sorted(perf_dep, key=lambda f: -mi_perf_all_dict[f]):
        print(f'     → {FEAT_LABELS[f]}: MI_perf={mi_perf_all_dict[f]:.4f}')

    # ── cell 72 ───────────────────────────────────────────────────
    from sklearn.metrics import roc_auc_score

    def pairwise_auroc(df, feature, svara_list):
        """AUROC matrix for every pair of svaras using `feature`."""
        n   = len(svara_list)
        mat = np.full((n, n), np.nan)
        for i, sv_i in enumerate(svara_list):
            for j, sv_j in enumerate(svara_list):
                if i >= j:
                    continue
                mask  = df['svara_label'].isin([sv_i, sv_j])
                sub   = df[mask]
                y     = (sub['svara_label'] == sv_i).astype(int)
                x     = sub[feature]
                valid = x.notna()
                if valid.sum() < 10:
                    continue
                auc = roc_auc_score(y[valid], x[valid])
                auc = max(auc, 1.0 - auc)   # always >= 0.5
                mat[i, j] = auc
                mat[j, i] = auc
        return mat

    auroc_matrices = {f: pairwise_auroc(df_pd, f, svara_labels) for f in ALL_FEATS}
    print(f'AUROC computed for {len(auroc_matrices)} descriptors × {len(svara_labels)} svaras')

    # ── cell 73 ───────────────────────────────────────────────────
    import seaborn as sns

    n_feats = len(ALL_FEATS)
    n_cols  = 4
    n_rows  = (n_feats + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.8, n_rows * 3.4))
    axes = axes.flatten()

    for k, feat in enumerate(ALL_FEATS):
        ax  = axes[k]
        mat = auroc_matrices[feat]
        sns.heatmap(
            mat,
            xticklabels=svara_labels,
            yticklabels=svara_labels,
            vmin=0.5, vmax=1.0,
            cmap='RdYlGn',
            annot=True, fmt='.2f',
            linewidths=0.4,
            ax=ax,
            cbar=False,
            annot_kws={'size': 7},
        )
        ax.set_title(FEAT_LABELS.get(feat, feat), fontsize=8, pad=4)
        ax.tick_params(labelsize=7)

    for k in range(n_feats, len(axes)):
        axes[k].set_visible(False)

    plt.suptitle(
        'Pairwise AUROC per descriptor  (verd = bona discriminació, vermell = cap)',
        fontsize=11, y=1.01
    )
    plt.tight_layout()
    _save(fig, "25_auroc_heatmaps", out_dir)

    # ── cell 74 ───────────────────────────────────────────────────
    # For each svara pair: which descriptor achieves the highest AUROC?
    import pandas as pd

    n = len(svara_labels)
    best_feat_mat  = np.empty((n, n), dtype=object)
    best_auroc_mat = np.full((n, n), np.nan)
    best_feat_mat[:] = '—'

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            best_auc  = 0.5
            best_feat = '—'
            for feat in ALL_FEATS:
                val = auroc_matrices[feat][i, j]
                if np.isfinite(val) and val > best_auc:
                    best_auc  = val
                    best_feat = feat
            best_feat_mat[i, j]  = best_feat
            best_auroc_mat[i, j] = best_auc

    df_best      = pd.DataFrame(best_feat_mat,         index=svara_labels, columns=svara_labels)
    df_best_auroc = pd.DataFrame(best_auroc_mat.round(2), index=svara_labels, columns=svara_labels)

    fig, axes = plt.subplots(1, 2, figsize=(16, 4))

    # Left: best AUROC value per pair
    sns.heatmap(
        best_auroc_mat,
        xticklabels=svara_labels, yticklabels=svara_labels,
        vmin=0.5, vmax=1.0, cmap='RdYlGn',
        annot=True, fmt='.2f',
        linewidths=0.4, ax=axes[0],
    )
    axes[0].set_title('Millor AUROC per parella (màxim sobre tots els descriptors)')

    # Right: which descriptor achieves that best AUROC
    # Encode feature names as integers for colour, annotate with short name
    feat_idx = {f: i for i, f in enumerate(ALL_FEATS)}
    code_mat = np.array([[feat_idx.get(best_feat_mat[i, j], -1) for j in range(n)] for i in range(n)], dtype=float)
    np.fill_diagonal(code_mat, np.nan)

    short = {f: FEAT_LABELS.get(f, f).split(']')[-1].strip()[:14] for f in ALL_FEATS}
    annot_mat = np.array([[short.get(best_feat_mat[i, j], '—') for j in range(n)] for i in range(n)])

    sns.heatmap(
        code_mat,
        xticklabels=svara_labels, yticklabels=svara_labels,
        cmap='tab20', annot=annot_mat, fmt='',
        linewidths=0.4, ax=axes[1], cbar=False,
        annot_kws={'size': 7},
    )
    axes[1].set_title('Descriptor amb millor AUROC per parella')

    plt.tight_layout()
    _save(fig, "fig_74", out_dir)




def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="v1", help="Output subfolder tag")
    args = parser.parse_args()

    out_dir = S.FIGURES_DIR / "structural_analysis" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[svara_mi_analysis] tag={args.tag!r}  output → {out_dir}")

    _, df, svara_labels, performers = load_all()
    run_mi_analysis(df, svara_labels, performers, out_dir)


if __name__ == "__main__":
    main()
