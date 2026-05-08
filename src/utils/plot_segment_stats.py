"""
Segment distribution plots for svara structural analysis.

Generates plots 01–12 (per-performer) and 15–19 (combined all-performers).

Usage:
    python -m src.utils.plot_segment_stats --tag v1_TR
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
import matplotlib.cm as cm
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import settings as S
from src.analysis.svara_segment_analysis import load_all, COLORS


def _save(fig, name: str, out_dir: Path, dpi: int = 120) -> None:
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path.name}")


def plot_segment_stats(all_rows, df, svara_labels, performers, out_dir: Path) -> None:
    """Generate all segment distribution and combined plots."""
    n_s = len(svara_labels)
    n_p = len(performers)
    xi  = np.arange(n_s)
    SVARA_COLORS = {sl: plt.cm.tab10(i / max(len(svara_labels), 1))
                    for i, sl in enumerate(svara_labels)}
    patches = [mpatches.Patch(color=COLORS.get(p, 'gray'), label=p, alpha=0.85) for p in performers]

    # ── cell 12 ───────────────────────────────────────────────────
    # Agregació: per (svara_label, performer) suma les durades
    agg_rows = []
    for sl in svara_labels:
        for perf in performers:
            sub = df.filter((pl.col('svara_label') == sl) & (pl.col('performer') == perf))
            if sub.is_empty():
                continue
            total_svara = float(sub['svara_dur_sec'].sum())
            total_cp    = float(sub['cp_total_dur_sec'].sum())
            total_sta   = float(sub['sta_total_dur_sec'].sum())
            total_tr    = float(sub['tr_total_dur_sec'].sum())
            total_sil   = float(sub['sil_total_dur_sec'].sum())
            agg_rows.append({
                'svara_label': sl,
                'performer':   perf,
                'total_svara': total_svara,
                'total_cp':    total_cp,
                'total_sta':   total_sta,
                'total_tr':    total_tr,
                'total_sil':   total_sil,
                'cp_frac':     total_cp  / total_svara if total_svara > 0 else 0.0,
                'sta_frac':    total_sta / total_svara if total_svara > 0 else 0.0,
                'tr_frac':     total_tr  / total_svara if total_svara > 0 else 0.0,
                'sil_frac':    total_sil / total_svara if total_svara > 0 else 0.0,
            })

    df_agg = pl.DataFrame(agg_rows)

    def agg_val(sl, perf, col):
        sub = df_agg.filter((pl.col('svara_label') == sl) & (pl.col('performer') == perf))
        return float(sub[col][0]) if not sub.is_empty() else 0.0

    # ── cell 14 ───────────────────────────────────────────────────
    bar_w = 0.8 / n_p
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Duració total STA
    ax = axes[0]
    for pi, perf in enumerate(performers):
        vals = [agg_val(sl, perf, 'total_sta') for sl in svara_labels]
        offset = (pi - n_p / 2 + 0.5) * bar_w
        ax.bar(xi + offset, vals, width=bar_w, color=COLORS.get(perf, 'gray'), label=perf, alpha=0.85)
    ax.set_xticks(xi)
    ax.set_xticklabels(svara_labels, fontsize=13, fontweight='bold')
    ax.set_ylabel('Duració total STA (s)')
    ax.set_title('Duració total STA per svara (suma de totes les ocurrències)')
    ax.legend(handles=patches, fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Fracció STA
    ax = axes[1]
    for pi, perf in enumerate(performers):
        vals = [agg_val(sl, perf, 'sta_frac') for sl in svara_labels]
        offset = (pi - n_p / 2 + 0.5) * bar_w
        ax.bar(xi + offset, vals, width=bar_w, color=COLORS.get(perf, 'gray'), label=perf, alpha=0.85)
    ax.set_xticks(xi)
    ax.set_xticklabels(svara_labels, fontsize=13, fontweight='bold')
    ax.set_ylabel('Fracció STA (STA / durada total svara)')
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, ls='--', color='gray', lw=0.8)
    ax.set_title('Fracció STA per svara (STA total / durada total)')
    ax.legend(handles=patches, fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    fig.suptitle('STA: duració i fracció per svara i intèrpret', fontsize=13)
    plt.tight_layout()
    _save(fig, "01_cp_duration", out_dir)

    # ── cell 16 ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, n_p, figsize=(3.5 * n_p, 5), sharey=True)
    xi = np.arange(n_s)

    for ax, perf in zip(axes, performers):
        cp_fracs  = [agg_val(sl, perf, 'cp_frac')  for sl in svara_labels]
        sta_fracs = [agg_val(sl, perf, 'sta_frac') for sl in svara_labels]
        tr_fracs  = [agg_val(sl, perf, 'tr_frac')  for sl in svara_labels]
        sil_fracs = [agg_val(sl, perf, 'sil_frac') for sl in svara_labels]

        bot_sta = cp_fracs
        bot_tr  = [c + s for c, s in zip(cp_fracs, sta_fracs)]
        bot_sil = [c + s + t for c, s, t in zip(cp_fracs, sta_fracs, tr_fracs)]

        ax.bar(xi, cp_fracs,  label='CP',  color='#4caf50',  alpha=0.85)
        ax.bar(xi, sta_fracs, label='STA', color='#e91e8c',  alpha=0.85, bottom=bot_sta)
        ax.bar(xi, tr_fracs,  label='TR',  color='#ff9800',  alpha=0.85, bottom=bot_tr)
        ax.bar(xi, sil_fracs, label='SIL', color='#9e9e9e',  alpha=0.85, bottom=bot_sil)

        ax.set_xticks(xi)
        ax.set_xticklabels(svara_labels, fontsize=12, fontweight='bold')
        ax.set_title(perf, fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.axhline(1.0, color='black', lw=0.5, ls='--', alpha=0.4)
        if ax == axes[0]:
            ax.set_ylabel('Fracció del temps total', fontsize=10)
            ax.legend(loc='upper right', fontsize=9)

    fig.suptitle('CP + STA + TR + SIL — fracció per svara i intèrpret', fontsize=12, fontweight='bold')
    fig.tight_layout()
    _save(fig, "02_stacked_fractions", out_dir)

    # ── cell 18 ───────────────────────────────────────────────────
    from scipy.signal import find_peaks

    fig, axes = plt.subplots(1, n_s, figsize=(3.5 * n_s, 5), sharey=False)
    if n_s == 1:
        axes = [axes]

    for ax, sl in zip(axes, svara_labels):
        for perf in performers:
            cents_vals = []
            for r in all_rows:
                if r['svara_label'] == sl and r['performer'] == perf:
                    cents_vals.extend(r['cp_cents_list'])
            if len(cents_vals) < 3:
                continue
            vals = np.array(cents_vals)
            vals = vals[np.isfinite(vals)]
            if len(vals) < 3:
                continue
            try:
                kde  = gaussian_kde(vals, bw_method=0.3)
                x    = np.linspace(vals.min() - 60, vals.max() + 60, 600)
                y    = kde(x)
                ax.plot(x, y, color=COLORS.get(perf, 'gray'), label=perf, linewidth=2)

                # Màxims locals: alçada ≥ 15% del pic global, separació ≥ 30 punts (~30 cents)
                peaks, _ = find_peaks(y, height=0.15 * y.max(), distance=30)
                for pk in peaks:
                    ax.axvline(x[pk], color=COLORS.get(perf, 'gray'), ls=':', lw=1.5, alpha=0.85)
                    ax.scatter(x[pk], y[pk], color=COLORS.get(perf, 'gray'),
                               s=45, zorder=5, marker='^', edgecolors='black', linewidths=0.5)
            except Exception:
                pass

        ax.set_title(sl, fontsize=14, fontweight='bold')
        ax.set_xlabel('Cents (relatiu a Sa)')
        ax.set_ylabel('Densitat')
        ax.axvline(0, color='black', lw=0.8, alpha=0.4)

    fig.legend(handles=patches, loc='upper right', fontsize=9)
    fig.suptitle('Distribució de cents CP per svara i intèrpret  (▲ = màxim local KDE)', fontsize=13)
    plt.tight_layout()
    _save(fig, "03_kde_cp_cents", out_dir)

    # ── cell 20 ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, n_s, figsize=(3.5 * n_s, 5), sharey=False)
    if n_s == 1:
        axes = [axes]

    for ax, sl in zip(axes, svara_labels):
        for perf in performers:
            all_sta, peaks_c, valleys_c = [], [], []
            for r in all_rows:
                if r['svara_label'] == sl and r['performer'] == perf:
                    all_sta.extend(r['sta_cents_list'])
                    peaks_c.extend(r['sta_peak_cents_list'])
                    valleys_c.extend(r['sta_valley_cents_list'])

            vals = np.array(all_sta)
            vals = vals[np.isfinite(vals)]
            if len(vals) < 3:
                continue

            color = COLORS.get(perf, 'gray')
            x     = np.linspace(vals.min() - 60, vals.max() + 60, 600)

            # histogram (density) behind KDE
            ax.hist(vals, bins=25, density=True, alpha=0.10, color=color, edgecolor='none')

            try:
                y = gaussian_kde(vals, bw_method=0.3)(x)
                ax.plot(x, y, color=color, label=perf, linewidth=2)
                peaks, _ = find_peaks(y, height=0.15 * y.max(), distance=30)
                for pk in peaks:
                    ax.axvline(x[pk], color=color, ls=':', lw=1.5, alpha=0.85)
                    ax.scatter(x[pk], y[pk], color=color, s=45, zorder=5,
                               marker='^', edgecolors='black', linewidths=0.5)
            except Exception:
                pass

            # peak-type STA (-- dashed, scaled to fraction of total)
            p_arr = np.array(peaks_c)[np.isfinite(np.array(peaks_c, dtype=float))]
            if len(p_arr) >= 3:
                try:
                    yp = gaussian_kde(p_arr, bw_method=0.3)(x) * (len(p_arr) / len(vals))
                    ax.plot(x, yp, color=color, ls='--', lw=1.2, alpha=0.6)
                except Exception:
                    pass

            # valley-type STA (dotted)
            v_arr = np.array(valleys_c)[np.isfinite(np.array(valleys_c, dtype=float))]
            if len(v_arr) >= 3:
                try:
                    yv = gaussian_kde(v_arr, bw_method=0.3)(x) * (len(v_arr) / len(vals))
                    ax.plot(x, yv, color=color, ls=':', lw=1.2, alpha=0.6)
                except Exception:
                    pass

        ax.set_title(sl, fontsize=14, fontweight='bold')
        ax.set_xlabel('Cents (relatiu a Sa)')
        ax.set_ylabel('Densitat')
        ax.axvline(0, color='black', lw=0.8, alpha=0.4)

    fig.legend(handles=patches, loc='upper right', fontsize=9)
    fig.suptitle('[D] STA cents — histogram + KDE per svara i intèrpret\n'
                 '▲=màxim KDE  ——=peaks (above CP)  ⋯=valleys (below CP)', fontsize=12)
    plt.tight_layout()
    _save(fig, "04_kde_sta_cents", out_dir)

    # ── cell 22 ───────────────────────────────────────────────────
    COLOR_PEAK   = '#e6550d'   # orange = above CP (ornamental peak)
    COLOR_VALLEY = '#3182bd'   # blue   = below CP (ornamental valley / dip)

    fig, axes = plt.subplots(1, n_s, figsize=(3.5 * n_s, 5), sharey=False)
    if n_s == 1:
        axes = [axes]

    for ax, sl in zip(axes, svara_labels):
        all_peaks, all_valleys = [], []
        for r in all_rows:
            if r['svara_label'] == sl:
                all_peaks.extend(r['sta_peak_cents_list'])
                all_valleys.extend(r['sta_valley_cents_list'])

        p_arr = np.array(all_peaks,   dtype=float)
        v_arr = np.array(all_valleys, dtype=float)
        p_arr = p_arr[np.isfinite(p_arr)]
        v_arr = v_arr[np.isfinite(v_arr)]

        all_vals = np.concatenate([p_arr, v_arr])
        if len(all_vals) == 0:
            continue
        bins = np.linspace(all_vals.min() - 30, all_vals.max() + 30, 35)

        if len(p_arr):
            ax.hist(p_arr, bins=bins, density=False, color=COLOR_PEAK,
                    alpha=0.65, edgecolor='white', linewidth=0.4, label='Peak (above CP)')
        if len(v_arr):
            ax.hist(v_arr, bins=bins, density=False, color=COLOR_VALLEY,
                    alpha=0.65, edgecolor='white', linewidth=0.4, label='Valley (below CP)')

        # KDE overlays
        x = np.linspace(all_vals.min() - 60, all_vals.max() + 60, 500)
        scale_p = len(p_arr) * (bins[1] - bins[0])
        scale_v = len(v_arr) * (bins[1] - bins[0])
        try:
            if len(p_arr) >= 3:
                ax.plot(x, gaussian_kde(p_arr, bw_method=0.3)(x) * scale_p,
                        color=COLOR_PEAK, lw=2)
            if len(v_arr) >= 3:
                ax.plot(x, gaussian_kde(v_arr, bw_method=0.3)(x) * scale_v,
                        color=COLOR_VALLEY, lw=2)
        except Exception:
            pass

        ax.axvline(0, color='black', lw=0.8, alpha=0.4)
        ax.set_title(sl, fontsize=14, fontweight='bold')
        ax.set_xlabel('Cents (relatiu a Sa)')
        ax.set_ylabel('Compte')
        n_peaks, n_valleys = len(p_arr), len(v_arr)
        ax.text(0.97, 0.97, f'peaks={n_peaks}\nvalleys={n_valleys}',
                transform=ax.transAxes, ha='right', va='top', fontsize=8, color='gray')

    axes[0].legend(fontsize=9)
    fig.suptitle('[D] STA peak vs valley — histogram per svara, tots els intèrprets combinats\n'
                 'Peak = STA amb pitch > mean CP  |  Valley = STA amb pitch ≤ mean CP', fontsize=12)
    plt.tight_layout()
    _save(fig, "05_sta_peak_valley_hist", out_dir)

    # ── cell 24 ───────────────────────────────────────────────────
    df_summary = (
        df
        .group_by(['svara_label', 'performer'])
        .agg([
            pl.col('svara_label').count().alias('n_occ'),
            pl.col('svara_dur_sec').mean().alias('mean_svara_dur'),
            pl.col('cp_total_dur_sec').mean().alias('mean_cp_dur'),
            pl.col('cp_frac').mean().alias('mean_cp_frac'),
            pl.col('sta_total_dur_sec').mean().alias('mean_sta_dur'),
            pl.col('sta_frac').mean().alias('mean_sta_frac'),
            pl.col('tr_total_dur_sec').mean().alias('mean_tr_dur'),
            pl.col('tr_frac').mean().alias('mean_tr_frac'),
            pl.col('sil_frac').mean().alias('mean_sil_frac'),
            pl.col('cp_mean_cents').drop_nans().mean().alias('mean_cp_cents'),
            pl.col('n_sta').mean().alias('mean_n_sta'),
            pl.col('sta_mean_cents').drop_nans().mean().alias('mean_sta_cents'),
        ])
        .sort(['svara_label', 'performer'])
    )

    metrics = [
        ('mean_cp_frac',  'Fracció CP',  'Blues',   False),
        ('mean_sta_frac', 'Fracció STA', 'Oranges', False),
        ('mean_tr_frac',  'Fracció TR',  'YlOrBr',  False),
        ('mean_sil_frac', 'Fracció SIL', 'Greys',   False),
        ('mean_cp_cents', 'CP cents',    'RdYlGn',  True),
        ('mean_sta_cents','STA cents',   'PuOr',    True),
    ]

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(3.5 * n_metrics, 4))

    for ax, (col, title, cmap, diverging) in zip(axes, metrics):
        pivot = (
            df_summary
            .select(['svara_label', 'performer', col])
            .pivot(index='svara_label', on='performer', values=col)
            .sort('svara_label')
        )
        data = pivot.select([c for c in pivot.columns if c != 'svara_label']).to_numpy().astype(float)
        im = ax.imshow(data, aspect='auto', cmap=cmap)
        ax.set_xticks(range(len(performers)))
        ax.set_xticklabels(performers, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(svara_labels)))
        ax.set_yticklabels(svara_labels, fontsize=9)
        ax.set_title(title, fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle('Resum per svara × intèrpret', fontsize=11, fontweight='bold')
    fig.tight_layout()
    _save(fig, "06_heatmap_summary", out_dir)

    # ── cell 27 ───────────────────────────────────────────────────
    # CP individual: 1 punt = 1 segment CP
    fig, ax = plt.subplots(figsize=(13, 5))

    for xi, sl in enumerate(svara_labels):
        for pi, perf in enumerate(performers):
            durs = []
            for r in all_rows:
                if r['svara_label'] == sl and r['performer'] == perf:
                    durs.extend(r['cp_dur_list'])
            if not durs:
                continue
            vals = np.array(durs)
            vals = vals[np.isfinite(vals) & (vals > 0)]
            if len(vals) == 0:
                continue

            x_off = xi + (pi - (n_p - 1) / 2) * 0.13
            jitter = np.random.default_rng(hash(perf + 'cp_ind_dur') % 2**32).uniform(-0.04, 0.04, size=len(vals))
            ax.scatter(x_off + jitter, vals, color=COLORS.get(perf, 'gray'), alpha=0.35, s=10, linewidths=0)

            med = np.median(vals)
            m   = np.mean(vals)
            s   = np.std(vals)
            ax.scatter(x_off, med, color=COLORS.get(perf, 'gray'), s=70, zorder=5,
                       edgecolors='black', linewidths=0.8, marker='o')
            ax.errorbar(x_off, m, yerr=s, fmt='D', color=COLORS.get(perf, 'gray'),
                        markersize=5, capsize=3, capthick=1.5, elinewidth=1.5, zorder=6,
                        markeredgecolor='black', markeredgewidth=0.5)

    ax.set_xticks(range(len(svara_labels)))
    ax.set_xticklabels(svara_labels, fontsize=13, fontweight='bold')
    ax.set_xlim(-0.5, n_s - 0.5)
    ax.set_ylabel('Durada segment CP (s)')
    ax.set_title('Durada de cada segment CP per svara  —  1 punt = 1 segment CP individual\n'
                 '● mediana  ◆ ± barres: mitjana ± std', fontsize=11)
    ax.legend(handles=patches, loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    _save(fig, "07_cp_individual_scatter", out_dir)

    # ── cell 29 ───────────────────────────────────────────────────
    # CP per ocurrència: max run consecutiva de CP per svara anotada
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for ax, col, ylabel, title_suffix in [
        (axes[0], 'cp_run_max_dur',  'Durada màx. run CP consecutius (s)', 'durada màx. run CP'),
        (axes[1], 'cp_run_max_frac', 'Fracció màx. run CP (/ durada svara)', 'fracció màx. run CP'),
    ]:
        for xi, sl in enumerate(svara_labels):
            for pi, perf in enumerate(performers):
                sub  = df.filter((pl.col('svara_label') == sl) & (pl.col('performer') == perf))
                vals = sub[col].to_numpy()
                vals = vals[np.isfinite(vals)]
                if len(vals) == 0:
                    continue

                x_off = xi + (pi - (n_p - 1) / 2) * 0.13
                jitter = np.random.default_rng(hash(perf + col) % 2**32).uniform(-0.04, 0.04, size=len(vals))
                ax.scatter(x_off + jitter, vals, color=COLORS.get(perf, 'gray'), alpha=0.5, s=15, linewidths=0)

                med = np.median(vals)
                m   = np.mean(vals)
                s   = np.std(vals)
                ax.scatter(x_off, med, color=COLORS.get(perf, 'gray'), s=70, zorder=5,
                           edgecolors='black', linewidths=0.8, marker='o')
                ax.errorbar(x_off, m, yerr=s, fmt='D', color=COLORS.get(perf, 'gray'),
                            markersize=5, capsize=3, capthick=1.5, elinewidth=1.5, zorder=6,
                            markeredgecolor='black', markeredgewidth=0.5)

        ax.set_xticks(range(len(svara_labels)))
        ax.set_xticklabels(svara_labels, fontsize=13, fontweight='bold')
        ax.set_xlim(-0.5, n_s - 0.5)
        ax.set_ylabel(ylabel)
        ax.set_title(f'CP per svara anotat — {title_suffix}\n'
                     '1 punt = 1 svara  (max run CP successius)  |  ● mediana  ◆ ± barres: mitjana ± std', fontsize=10)
        ax.legend(handles=patches, loc='upper right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        if col == 'cp_run_max_frac':
            ax.set_ylim(0, 1.05)
            ax.axhline(0.5, ls='--', color='gray', lw=0.8)

    fig.suptitle('CP per svara anotat — màx. run consecutiva de CP', fontsize=13)
    plt.tight_layout()
    _save(fig, "08_cp_occurrence_scatter", out_dir)

    # ── cell 31 ───────────────────────────────────────────────────
    # STA+TR per ocurrència: max run consecutiva de STA+TR (qualsevol combinació)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for ax, col, ylabel, title_suffix in [
        (axes[0], 'sta_tr_run_max_dur',  'Durada màx. run STA+TR consecutius (s)', 'durada màx. run STA+TR'),
        (axes[1], 'sta_tr_run_max_frac', 'Fracció màx. run STA+TR (/ durada svara)', 'fracció màx. run STA+TR'),
    ]:
        for xi, sl in enumerate(svara_labels):
            for pi, perf in enumerate(performers):
                sub  = df.filter((pl.col('svara_label') == sl) & (pl.col('performer') == perf))
                vals = sub[col].to_numpy()
                vals = vals[np.isfinite(vals)]
                if len(vals) == 0:
                    continue

                x_off = xi + (pi - (n_p - 1) / 2) * 0.13
                jitter = np.random.default_rng(hash(perf + col) % 2**32).uniform(-0.04, 0.04, size=len(vals))
                ax.scatter(x_off + jitter, vals, color=COLORS.get(perf, 'gray'), alpha=0.5, s=15, linewidths=0)

                med = np.median(vals)
                m   = np.mean(vals)
                s   = np.std(vals)
                ax.scatter(x_off, med, color=COLORS.get(perf, 'gray'), s=70, zorder=5,
                           edgecolors='black', linewidths=0.8, marker='o')
                ax.errorbar(x_off, m, yerr=s, fmt='D', color=COLORS.get(perf, 'gray'),
                            markersize=5, capsize=3, capthick=1.5, elinewidth=1.5, zorder=6,
                            markeredgecolor='black', markeredgewidth=0.5)

        ax.set_xticks(range(len(svara_labels)))
        ax.set_xticklabels(svara_labels, fontsize=13, fontweight='bold')
        ax.set_xlim(-0.5, n_s - 0.5)
        ax.set_ylabel(ylabel)
        ax.set_title(f'STA+TR per svara anotat — {title_suffix}\n'
                     '1 punt = 1 svara  (max run STA+TR successius)  |  ● mediana  ◆ ± barres: mitjana ± std', fontsize=10)
        ax.legend(handles=patches, loc='upper right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        if col == 'sta_tr_run_max_frac':
            ax.set_ylim(0, 1.05)
            ax.axhline(0.5, ls='--', color='gray', lw=0.8)

    fig.suptitle('STA+TR per svara anotat — màx. run consecutiva de STA+TR', fontsize=13)
    plt.tight_layout()
    _save(fig, "09_sta_tr_occurrence_scatter", out_dir)

    # ── cell 33 ───────────────────────────────────────────────────
    # STA per ocurrència: 1 punt = 1 svara anotada (suma dels STA interns)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for ax, col, ylabel, title_suffix in [
        (axes[0], 'sta_total_dur_sec', 'Durada total STA per svara (s)', 'durada total STA'),
        (axes[1], 'sta_frac',          'Fracció STA (STA total / durada svara)', 'fracció STA'),
    ]:
        for xi, sl in enumerate(svara_labels):
            for pi, perf in enumerate(performers):
                sub  = df.filter((pl.col('svara_label') == sl) & (pl.col('performer') == perf))
                vals = sub[col].to_numpy()
                vals = vals[np.isfinite(vals)]
                if len(vals) == 0:
                    continue

                x_off = xi + (pi - (n_p - 1) / 2) * 0.13
                jitter = np.random.default_rng(hash(perf + col + 'sta') % 2**32).uniform(-0.04, 0.04, size=len(vals))
                ax.scatter(x_off + jitter, vals, color=COLORS.get(perf, 'gray'), alpha=0.5, s=15, linewidths=0)

                med = np.median(vals)
                m   = np.mean(vals)
                s   = np.std(vals)
                ax.scatter(x_off, med, color=COLORS.get(perf, 'gray'), s=70, zorder=5,
                           edgecolors='black', linewidths=0.8, marker='o')
                ax.errorbar(x_off, m, yerr=s, fmt='D', color=COLORS.get(perf, 'gray'),
                            markersize=5, capsize=3, capthick=1.5, elinewidth=1.5, zorder=6,
                            markeredgecolor='black', markeredgewidth=0.5)

        ax.set_xticks(range(len(svara_labels)))
        ax.set_xticklabels(svara_labels, fontsize=13, fontweight='bold')
        ax.set_xlim(-0.5, n_s - 0.5)
        ax.set_ylabel(ylabel)
        ax.set_title(f'STA per svara anotat — {title_suffix}\n'
                     '1 punt = 1 svara  (suma STA interns)  |  ● mediana  ◆ ± barres: mitjana ± std', fontsize=10)
        ax.legend(handles=patches, loc='upper right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        if col == 'sta_frac':
            ax.set_ylim(0, 1.05)
            ax.axhline(0.5, ls='--', color='gray', lw=0.8)

    fig.suptitle('STA per svara anotat — durada i fracció', fontsize=13)
    plt.tight_layout()
    _save(fig, "10_sta_occurrence_scatter", out_dir)

    # ── cell 35 ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for ax, cents_key, seg_type in [
        (axes[0], 'cp_cents_list',  'CP'),
        (axes[1], 'sta_cents_list', 'STA'),
    ]:
        for xi, sl in enumerate(svara_labels):
            for pi, perf in enumerate(performers):
                cents_vals = []
                for r in all_rows:
                    if r['svara_label'] == sl and r['performer'] == perf:
                        cents_vals.extend(r[cents_key])
                vals = np.array(cents_vals, dtype=float)
                vals = vals[np.isfinite(vals)]
                if len(vals) == 0:
                    continue

                x_off  = xi + (pi - (n_p - 1) / 2) * 0.13
                jitter = np.random.default_rng(hash(perf + cents_key + sl) % 2**32).uniform(-0.04, 0.04, size=len(vals))
                ax.scatter(x_off + jitter, vals, color=COLORS.get(perf, 'gray'), alpha=0.3, s=8, linewidths=0)

                med, m, s = np.median(vals), np.mean(vals), np.std(vals)
                ax.scatter(x_off, med, color=COLORS.get(perf, 'gray'), s=70, zorder=5,
                           edgecolors='black', linewidths=0.8, marker='o')
                ax.errorbar(x_off, m, yerr=s, fmt='D', color=COLORS.get(perf, 'gray'),
                            markersize=5, capsize=3, capthick=1.5, elinewidth=1.5, zorder=6,
                            markeredgecolor='black', markeredgewidth=0.5)

        ax.axhline(0, color='black', lw=0.8, alpha=0.4)
        ax.set_xticks(range(len(svara_labels)))
        ax.set_xticklabels(svara_labels, fontsize=13, fontweight='bold')
        ax.set_xlim(-0.5, n_s - 0.5)
        ax.set_ylabel(f'Cents {seg_type} (relatiu a Sa)')
        ax.set_title(f'Cents {seg_type} per svara i intèrpret  —  1 punt = 1 segment {seg_type}\n'
                     '● mediana  ◆± barres: mitjana ± std', fontsize=10)
        ax.legend(handles=patches, loc='upper right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Cents CP i STA per svara i intèrpret', fontsize=13)
    plt.tight_layout()
    _save(fig, "11_cents_cp_sta_scatter", out_dir)

    # ── cell 37 ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for ax, run_key, seg_type in [
        (axes[0], 'cp_run_dur_list',  'CP'),
        (axes[1], 'sta_run_dur_list', 'STA'),
    ]:
        for xi, sl in enumerate(svara_labels):
            for pi, perf in enumerate(performers):
                run_durs = []
                for r in all_rows:
                    if r['svara_label'] == sl and r['performer'] == perf:
                        run_durs.extend(r[run_key])
                vals = np.array(run_durs, dtype=float)
                vals = vals[np.isfinite(vals) & (vals > 0)]
                if len(vals) == 0:
                    continue

                x_off  = xi + (pi - (n_p - 1) / 2) * 0.13
                jitter = np.random.default_rng(hash(perf + run_key + sl) % 2**32).uniform(-0.04, 0.04, size=len(vals))
                ax.scatter(x_off + jitter, vals, color=COLORS.get(perf, 'gray'),
                           alpha=0.35, s=10, linewidths=0)
                med, m, s = np.median(vals), np.mean(vals), np.std(vals)
                ax.scatter(x_off, med, color=COLORS.get(perf, 'gray'), s=70, zorder=5,
                           edgecolors='black', linewidths=0.8, marker='o')
                ax.errorbar(x_off, m, yerr=s, fmt='D', color=COLORS.get(perf, 'gray'),
                            markersize=5, capsize=3, capthick=1.5, elinewidth=1.5, zorder=6,
                            markeredgecolor='black', markeredgewidth=0.5)

        ax.set_xticks(range(len(svara_labels)))
        ax.set_xticklabels(svara_labels, fontsize=13, fontweight='bold')
        ax.set_xlim(-0.5, n_s - 0.5)
        ax.set_ylabel(f'Durada del run {seg_type} (s)')
        ax.set_title(f'[C] {seg_type} run duration  —  1 punt = 1 contiguous {seg_type} block\n'
                     '● mediana  ◆± barres: mitjana ± std', fontsize=10)
        ax.legend(handles=patches, loc='upper right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('[C] Run duration per svara i intèrpret — CP i STA', fontsize=13)
    plt.tight_layout()
    _save(fig, "12_run_level", out_dir)

    # ── cell 48 ───────────────────────────────────────────────────
    def _scatter_combined(ax, vals_per_svara, ylabel, title, ylim=None, hline=None):
        """Scatter combinat (tots els intèrprets junts): punts + mediana + mitjana±std."""
        for xi, sl in enumerate(svara_labels):
            vals = np.array(vals_per_svara[sl])
            vals = vals[np.isfinite(vals) & (vals >= 0)]
            if len(vals) == 0:
                continue
            jitter = np.random.default_rng(hash(sl + title) % 2**32).uniform(-0.2, 0.2, size=len(vals))
            ax.scatter(xi + jitter, vals, color=SVARA_COLORS[sl], alpha=0.35, s=12, linewidths=0)
            med, m, s = np.median(vals), np.mean(vals), np.std(vals)
            ax.scatter(xi, med, color=SVARA_COLORS[sl], s=80, zorder=5,
                       edgecolors='black', linewidths=0.9, marker='o')
            ax.errorbar(xi, m, yerr=s, fmt='D', color=SVARA_COLORS[sl],
                        markersize=6, capsize=4, capthick=1.5, elinewidth=1.5, zorder=6,
                        markeredgecolor='black', markeredgewidth=0.6)
        ax.set_xticks(range(len(svara_labels)))
        ax.set_xticklabels(svara_labels, fontsize=13, fontweight='bold')
        ax.set_xlim(-0.5, n_s - 0.5)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        if ylim:
            ax.set_ylim(*ylim)
        if hline is not None:
            ax.axhline(hline, ls='--', color='gray', lw=0.8)

    # ── cell 49 ───────────────────────────────────────────────────
    # Prepara dicts: svara → llista de valors (de totes les gravacions)
    cp_ind_durs  = {sl: [] for sl in svara_labels}   # 1 valor per segment CP
    sta_ind_durs = {sl: [] for sl in svara_labels}   # 1 valor per segment STA
    tr_ind_durs  = {sl: [] for sl in svara_labels}   # 1 valor per segment TR
    cp_occ_durs  = {sl: [] for sl in svara_labels}   # 1 valor per svara anotada
    sta_occ_durs = {sl: [] for sl in svara_labels}
    tr_occ_durs  = {sl: [] for sl in svara_labels}
    cp_occ_frac  = {sl: [] for sl in svara_labels}
    sta_occ_frac = {sl: [] for sl in svara_labels}
    tr_occ_frac  = {sl: [] for sl in svara_labels}
    cp_cents_comb  = {sl: [] for sl in svara_labels}
    sta_cents_comb = {sl: [] for sl in svara_labels}

    for r in all_rows:
        sl = r['svara_label']
        if sl not in svara_labels:
            continue
        cp_ind_durs[sl].extend(r['cp_dur_list'])
        sta_ind_durs[sl].extend(r['sta_dur_list'])
        tr_ind_durs[sl].extend(r['tr_dur_list'])
        cp_occ_durs[sl].append(r['cp_total_dur_sec'])
        sta_occ_durs[sl].append(r['sta_total_dur_sec'])
        tr_occ_durs[sl].append(r['tr_total_dur_sec'])
        cp_occ_frac[sl].append(r['cp_frac'])
        sta_occ_frac[sl].append(r['sta_frac'])
        tr_occ_frac[sl].append(r['tr_frac'])
        cp_cents_comb[sl].extend(r['cp_cents_list'])
        sta_cents_comb[sl].extend(r['sta_cents_list'])

    # ── cell 50 ───────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    _scatter_combined(axes[0, 0], cp_ind_durs,
        ylabel='Durada segment CP (s)',
        title='CP individual  —  1 punt = 1 segment CP\n● mediana  ◆± barres: mitjana ± std')

    _scatter_combined(axes[0, 1], cp_occ_durs,
        ylabel='Durada total CP per svara (s)',
        title='CP per ocurrència  —  1 punt = 1 svara anotada (suma CP interns)\n● mediana  ◆± barres: mitjana ± std')

    _scatter_combined(axes[1, 0], sta_ind_durs,
        ylabel='Durada segment STA (s)',
        title='STA individual  —  1 punt = 1 segment STA\n● mediana  ◆± barres: mitjana ± std')

    _scatter_combined(axes[1, 1], sta_occ_durs,
        ylabel='Durada total STA per svara (s)',
        title='STA per ocurrència  —  1 punt = 1 svara anotada (suma STA interns)\n● mediana  ◆± barres: mitjana ± std')

    fig.suptitle('Durades CP i STA per svara — tots els intèrprets combinats', fontsize=14)
    plt.tight_layout()
    _save(fig, "15_combined_scatter", out_dir)

    # ── cell 51 ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    _scatter_combined(axes[0], cp_occ_frac,
        ylabel='Fracció CP (CP total / durada svara)',
        title='Fracció CP per svara  —  1 punt = 1 svara anotada\n● mediana  ◆± barres: mitjana ± std',
        ylim=(0, 1.05), hline=0.5)

    _scatter_combined(axes[1], sta_occ_frac,
        ylabel='Fracció STA (STA total / durada svara)',
        title='Fracció STA per svara  —  1 punt = 1 svara anotada\n● mediana  ◆± barres: mitjana ± std',
        ylim=(0, 1.05), hline=0.5)

    fig.suptitle('Fraccions CP i STA per svara — tots els intèrprets combinats', fontsize=14)
    plt.tight_layout()
    _save(fig, "16_combined_occ", out_dir)

    # ── cell 52 ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, cents_dict, label_type in [
        (axes[0], cp_cents_comb,  'CP'),
        (axes[1], sta_cents_comb, 'STA'),
    ]:
        for xi, sl in enumerate(svara_labels):
            vals = np.array(cents_dict[sl])
            vals = vals[np.isfinite(vals)]
            if len(vals) < 3:
                continue
            try:
                kde  = gaussian_kde(vals, bw_method=0.3)
                x    = np.linspace(vals.min() - 60, vals.max() + 60, 600)
                y    = kde(x)
                # Desplaça i escala la KDE per mostrar-la al costat de xi (violin-style horitzontal)
                # — o simplement la dibuixem com a subplot per svara —
                peaks, _ = find_peaks(y, height=0.15 * y.max(), distance=30)
                ax.plot(x, y + xi * 0.0, color=SVARA_COLORS[sl], linewidth=2, label=sl, alpha=0.85)
                for pk in peaks:
                    ax.axvline(x[pk], color=SVARA_COLORS[sl], ls=':', lw=1.5, alpha=0.8)
                    ax.scatter(x[pk], y[pk], color=SVARA_COLORS[sl], s=45, zorder=5,
                               marker='^', edgecolors='black', linewidths=0.5)
            except Exception:
                pass

        ax.axvline(0, color='black', lw=0.8, alpha=0.4)
        ax.set_xlabel('Cents (relatiu a Sa)')
        ax.set_ylabel('Densitat KDE')
        ax.set_title(f'Distribució cents {label_type} per svara\ntots els intèrprets combinats  (▲ = màxim local)', fontsize=11)
        ax.legend(title='Svara', fontsize=9)
        ax.grid(alpha=0.2)

    plt.tight_layout()
    _save(fig, "17_combined_cents_scatter", out_dir)

    # ── cell 53 ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, cents_dict, seg_type in [
        (axes[0], cp_cents_comb,  'CP'),
        (axes[1], sta_cents_comb, 'STA'),
    ]:
        for xi, sl in enumerate(svara_labels):
            vals = np.array(cents_dict[sl], dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue

            jitter = np.random.default_rng(hash(sl + seg_type + 'cents') % 2**32).uniform(-0.2, 0.2, size=len(vals))
            ax.scatter(xi + jitter, vals, color=SVARA_COLORS[sl], alpha=0.3, s=10, linewidths=0)

            med, m, s = np.median(vals), np.mean(vals), np.std(vals)
            ax.scatter(xi, med, color=SVARA_COLORS[sl], s=80, zorder=5,
                       edgecolors='black', linewidths=0.9, marker='o')
            ax.errorbar(xi, m, yerr=s, fmt='D', color=SVARA_COLORS[sl],
                        markersize=6, capsize=4, capthick=1.5, elinewidth=1.5, zorder=6,
                        markeredgecolor='black', markeredgewidth=0.6)

        ax.axhline(0, color='black', lw=0.8, alpha=0.4)
        ax.set_xticks(range(len(svara_labels)))
        ax.set_xticklabels(svara_labels, fontsize=13, fontweight='bold')
        ax.set_xlim(-0.5, n_s - 0.5)
        ax.set_ylabel(f'Cents {seg_type} (relatiu a Sa)')
        ax.set_title(f'Cents {seg_type}  —  1 punt = 1 segment {seg_type}\n'
                     'tots els intèrprets combinats  |  ● mediana  ◆± barres: mitjana ± std', fontsize=10)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Cents CP i STA per svara — tots els intèrprets combinats', fontsize=13)
    plt.tight_layout()
    _save(fig, "18_combined_kde", out_dir)

    # ── cell 55 ───────────────────────────────────────────────────
    cp_run_durs_comb  = {sl: [] for sl in svara_labels}
    sta_run_durs_comb = {sl: [] for sl in svara_labels}
    sta_peak_comb     = {sl: [] for sl in svara_labels}
    sta_valley_comb   = {sl: [] for sl in svara_labels}

    for r in all_rows:
        sl = r['svara_label']
        if sl not in svara_labels:
            continue
        cp_run_durs_comb[sl].extend(r['cp_run_dur_list'])
        sta_run_durs_comb[sl].extend(r['sta_run_dur_list'])
        sta_peak_comb[sl].extend(r['sta_peak_cents_list'])
        sta_valley_comb[sl].extend(r['sta_valley_cents_list'])

    # ── Run duration scatter ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, runs_dict, seg_type in [
        (axes[0], cp_run_durs_comb,  'CP'),
        (axes[1], sta_run_durs_comb, 'STA'),
    ]:
        for xi, sl in enumerate(svara_labels):
            vals = np.array(runs_dict[sl], dtype=float)
            vals = vals[np.isfinite(vals) & (vals > 0)]
            if len(vals) == 0:
                continue
            jitter = np.random.default_rng(hash(sl + seg_type + 'run') % 2**32).uniform(-0.2, 0.2, size=len(vals))
            ax.scatter(xi + jitter, vals, color=SVARA_COLORS[sl], alpha=0.3, s=10, linewidths=0)
            med, m, s = np.median(vals), np.mean(vals), np.std(vals)
            ax.scatter(xi, med, color=SVARA_COLORS[sl], s=80, zorder=5,
                       edgecolors='black', linewidths=0.9, marker='o')
            ax.errorbar(xi, m, yerr=s, fmt='D', color=SVARA_COLORS[sl],
                        markersize=6, capsize=4, capthick=1.5, elinewidth=1.5, zorder=6,
                        markeredgecolor='black', markeredgewidth=0.6)
        ax.set_xticks(range(len(svara_labels)))
        ax.set_xticklabels(svara_labels, fontsize=13, fontweight='bold')
        ax.set_xlim(-0.5, n_s - 0.5)
        ax.set_ylabel(f'Durada run {seg_type} (s)')
        ax.set_title(f'[C] {seg_type} run duration — combined\n1 punt = 1 contiguous block  |  ● mediana  ◆± std', fontsize=10)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('[C] Run duration CP i STA — tots els intèrprets combinats', fontsize=13)
    plt.tight_layout()
    _save(fig, "19_run_level_combined", out_dir)

    # ── STA peak vs valley histogram — combined ───────────────────────────────────
    fig, axes = plt.subplots(1, n_s, figsize=(3.5 * n_s, 4), sharey=False)
    if n_s == 1:
        axes = [axes]

    for ax, sl in zip(axes, svara_labels):
        p_arr = np.array(sta_peak_comb[sl],   dtype=float); p_arr = p_arr[np.isfinite(p_arr)]
        v_arr = np.array(sta_valley_comb[sl], dtype=float); v_arr = v_arr[np.isfinite(v_arr)]
        all_v = np.concatenate([p_arr, v_arr])
        if len(all_v) == 0:
            continue
        bins = np.linspace(all_v.min() - 30, all_v.max() + 30, 30)
        ax.hist(p_arr, bins=bins, color=COLOR_PEAK,   alpha=0.65, edgecolor='white', lw=0.4, label='Peak')
        ax.hist(v_arr, bins=bins, color=COLOR_VALLEY, alpha=0.65, edgecolor='white', lw=0.4, label='Valley')
        x = np.linspace(all_v.min() - 60, all_v.max() + 60, 400)
        bw = bins[1] - bins[0]
        try:
            if len(p_arr) >= 3:
                ax.plot(x, gaussian_kde(p_arr, bw_method=0.3)(x) * len(p_arr) * bw, color=COLOR_PEAK, lw=2)
            if len(v_arr) >= 3:
                ax.plot(x, gaussian_kde(v_arr, bw_method=0.3)(x) * len(v_arr) * bw, color=COLOR_VALLEY, lw=2)
        except Exception:
            pass
        ax.axvline(0, color='black', lw=0.8, alpha=0.4)
        ax.set_title(sl, fontsize=14, fontweight='bold')
        ax.set_xlabel('Cents')
        ax.set_ylabel('Compte')
        ax.text(0.97, 0.97, f'p={len(p_arr)}\nv={len(v_arr)}',
                transform=ax.transAxes, ha='right', va='top', fontsize=8, color='gray')

    axes[0].legend(fontsize=9)
    fig.suptitle('[D] STA peak vs valley — combined, tots els intèrprets', fontsize=12)
    plt.tight_layout()
    _save(fig, "19_run_level_combined", out_dir)



def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="v1", help="Output subfolder tag")
    args = parser.parse_args()

    out_dir = S.FIGURES_DIR / "structural_analysis" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[plot_segment_stats] tag={args.tag!r}  output → {out_dir}")

    all_rows, df, svara_labels, performers = load_all()
    plot_segment_stats(all_rows, df, svara_labels, performers, out_dir)


if __name__ == "__main__":
    main()
