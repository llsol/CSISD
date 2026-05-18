"""
Interactive pitch comparison between two extractors on a CV corpus recording.

Controls (right panel)
----------------------
  Source B       RadioButtons — switch between original / as / unet for extractor B
  Threshold (B)  CheckButtons — overlay multiple voiced thresholds simultaneously
                 (only active when extractor B file has 3 columns: time, f0_raw, conf)
  Save PNG       save the current view to figures/compare_pitch/

Usage
-----
  python -m src.pitch_extraction.compare_pitch srs_v1_svd_sav
  python -m src.pitch_extraction.compare_pitch srs_v1_svd_sav \\
      --extractor-a ftanet --extractor-b swiftf0scratch --source as
  python -m src.pitch_extraction.compare_pitch --all \\
      --extractor-a ftanet --extractor-b swiftf0scratch
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.widgets import RadioButtons, CheckButtons, Button
import numpy as np
from scipy import interpolate

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import settings

# ── constants ──────────────────────────────────────────────────────────────────

TOP_N_REGIONS   = 5
REGION_SEC      = 10.0
VOICED_MIN_HZ   = 50.0
TONIC_FALLBACK  = 200.0
AGREE_THR_CENTS = 50.0

AGREE_COLOR   = "#909090"
DISAG_A_COLOR = "#FF00CC"
DISAG_B_COLOR = "#00CCCC"
ONLY_A_COLOR  = "#8B0000"

THRESHOLD_LIST = [0.30, 0.50, 0.70, 0.80, 0.90, 0.95]
THR_COLORS     = {
    0.30: "#e41a1c",
    0.50: "#ff7f00",
    0.70: "#e6c200",
    0.80: "#4daf4a",
    0.90: "#377eb8",
    0.95: "#984ea3",
}
DEFAULT_THR = 0.90

EXTRACTOR_STYLES = {
    "ftanet":         ("FTA-Net",         "steelblue"),
    "swiftf0":        ("SwiftF0",         "tomato"),
    "swiftf0ft":      ("SwiftF0-ft",      "gold"),
    "swiftf0scratch": ("SwiftF0-scratch", "darkorange"),
}
EXTRACTOR_CV_DIR = {
    "ftanet":         "ftanet",
    "swiftf0":        "swiftf0",
    "swiftf0ft":      "swiftf0_finetune",
    "swiftf0scratch": "swiftf0_scratch",
}
EXTRACTOR_FILE_SUFFIX = {
    "ftanet":         "ftanet",
    "swiftf0":        "swiftf0",
    "swiftf0ft":      "swiftf0finetune",
    "swiftf0scratch": "swiftf0scratch",
}
SOURCE_LABELS = {
    "original": "original mix",
    "unet":     "U-Net separated",
    "as":       "BS-RoFormer separated",
}
ALL_SOURCES = ["original", "as", "unet"]


# ── data helpers ───────────────────────────────────────────────────────────────

def _pitch_path(extractor: str, source: str, recording_id: str) -> Path:
    cv_dir = EXTRACTOR_CV_DIR.get(extractor, extractor)
    suffix = EXTRACTOR_FILE_SUFFIX.get(extractor, extractor)
    return (settings.INTERIM_PITCH_CV / cv_dir / source / recording_id
            / f"{recording_id}_{source}_{suffix}_raw.npy")


def load_pitch(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Returns (times, f0_hz, conf_or_None). conf=None for 2-column pre-thresholded files."""
    data = np.load(path)
    if data.shape[1] >= 3:
        return data[:, 0], data[:, 1], data[:, 2]
    return data[:, 0], data[:, 1], None


def hz_to_cents(f0: np.ndarray, tonic: float) -> np.ndarray:
    out = np.full_like(f0, np.nan, dtype=float)
    voiced = f0 > VOICED_MIN_HZ
    out[voiced] = 1200.0 * np.log2(f0[voiced] / tonic)
    return out


def apply_thr(f0_hz: np.ndarray, conf: np.ndarray | None, thr: float) -> np.ndarray:
    if conf is None:
        return f0_hz
    out = f0_hz.copy()
    out[conf < thr] = 0.0
    return out


def align_cents(time_a, cents_a, time_b, cents_b):
    dt_a = np.median(np.diff(time_a)) if len(time_a) > 1 else 0.01
    dt_b = np.median(np.diff(time_b)) if len(time_b) > 1 else 0.016
    if abs(dt_a - dt_b) < 0.0005:
        n = min(len(time_a), len(time_b))
        return time_a[:n], cents_a[:n], cents_b[:n]
    if dt_a <= dt_b:
        cb = interpolate.interp1d(time_b, cents_b, kind="linear",
                                  bounds_error=False, fill_value=np.nan)(time_a)
        return time_a, cents_a, cb
    ca = interpolate.interp1d(time_a, cents_a, kind="linear",
                              bounds_error=False, fill_value=np.nan)(time_b)
    return time_b, ca, cents_b


def _top_divergence_regions(time, diff, n, window_sec):
    dt  = float(np.median(np.diff(time))) if len(time) > 1 else 0.01
    win = max(1, int(window_sec / dt))
    scores = []
    i = 0
    while i + win <= len(diff):
        v = diff[i: i + win]
        v = v[np.isfinite(v)]
        if len(v):
            scores.append((time[i], time[min(i + win - 1, len(time) - 1)],
                           np.mean(np.abs(v))))
        i += win
    scores.sort(key=lambda x: -x[2])
    selected: list = []
    for s, e, sc in scores:
        if all(e <= ps or s >= pe for ps, pe, _ in selected):
            selected.append((s, e, sc))
        if len(selected) == n:
            break
    return selected


def _masked_line(ax, t, v, mask, color, lw=0.9, zorder=2, **kw):
    ax.plot(t, np.where(mask, v, np.nan), color=color, lw=lw, zorder=zorder, **kw)


# ── interactive figure ─────────────────────────────────────────────────────────

class InteractiveFig:
    def __init__(self, recording_id, extractor_a, extractor_b, source_a):
        self.recording_id = recording_id
        self.extractor_a  = extractor_a
        self.extractor_b  = extractor_b
        self.source_a     = source_a
        self.label_a, self.color_a = EXTRACTOR_STYLES.get(extractor_a, (extractor_a, "steelblue"))
        self.label_b, self.color_b = EXTRACTOR_STYLES.get(extractor_b, (extractor_b, "tomato"))
        self.tonic = settings.RECORDING_SELECTION_TONICS.get(recording_id, TONIC_FALLBACK)

        # Load A (fixed source)
        path_a = _pitch_path(extractor_a, source_a, recording_id)
        if not path_a.exists():
            raise FileNotFoundError(f"A not found: {path_a}")
        t_a, f0_a, _ = load_pitch(path_a)
        self.time_a   = t_a
        self.cents_a  = hz_to_cents(f0_a, self.tonic)

        # Load B for all available sources
        self.data_b: dict[str, tuple] = {}
        for src in ALL_SOURCES:
            p = _pitch_path(extractor_b, src, recording_id)
            if p.exists():
                self.data_b[src] = load_pitch(p)

        if not self.data_b:
            raise FileNotFoundError(
                f"No predictions found for {extractor_b} on {recording_id}"
            )

        self.sources_b = list(self.data_b.keys())
        self.current_source_b = self.sources_b[0]

        # Threshold state: only meaningful when B has confidence (3-col files)
        self.has_conf_b = any(d[2] is not None for d in self.data_b.values())
        self.active_thrs: set[float] = {DEFAULT_THR} if self.has_conf_b else set()

        self._build_figure()
        self.update()

    # ── layout ────────────────────────────────────────────────────────────────

    def _build_figure(self):
        self.fig = plt.figure(figsize=(22, 11))
        self.fig.canvas.manager.set_window_title(
            f"{self.recording_id}  |  {self.label_a} vs {self.label_b}"
        )
        # Main grid (left 77% of figure)
        gs = self.fig.add_gridspec(
            3, 2, width_ratios=[4, 1],
            left=0.04, right=0.76, top=0.93, bottom=0.07,
            hspace=0.40, wspace=0.08,
        )
        self.ax0     = self.fig.add_subplot(gs[0, 0])
        self.ax1     = self.fig.add_subplot(gs[1, 0], sharex=self.ax0)
        self.ax2     = self.fig.add_subplot(gs[2, 0], sharex=self.ax0)
        self.ax_hist = self.fig.add_subplot(gs[:, 1])

        # ── widgets (right panel, figure coordinates) ──────────────────────────

        # Source B — RadioButtons
        n_src  = len(self.sources_b)
        h_rad  = 0.06 + n_src * 0.05
        ax_rad = self.fig.add_axes([0.78, 0.88 - h_rad, 0.20, h_rad])
        ax_rad.set_title("Source B", fontsize=9, pad=3)
        self._radio = RadioButtons(
            ax_rad, self.sources_b,
            active=self.sources_b.index(self.current_source_b),
        )
        self._radio.on_clicked(self._on_source)

        # Threshold CheckButtons (only if conf available)
        if self.has_conf_b:
            n_thr  = len(THRESHOLD_LIST)
            h_chk  = 0.06 + n_thr * 0.05
            ax_chk = self.fig.add_axes([0.78, 0.88 - h_rad - 0.06 - h_chk, 0.20, h_chk])
            ax_chk.set_title("Threshold (B)", fontsize=9, pad=3)
            labels  = [f"{t:.2f}" for t in THRESHOLD_LIST]
            actives = [t in self.active_thrs for t in THRESHOLD_LIST]
            self._check = CheckButtons(ax_chk, labels, actives)
            self._check.on_clicked(self._on_threshold)
            btn_y = 0.88 - h_rad - 0.06 - h_chk - 0.09
        else:
            ax_info = self.fig.add_axes([0.78, 0.45, 0.20, 0.12])
            ax_info.axis("off")
            ax_info.text(
                0.5, 0.5,
                "Threshold N/A\n(2-col file)\nRe-run predict\nto enable",
                ha="center", va="center", fontsize=8, color="gray",
                transform=ax_info.transAxes,
            )
            btn_y = 0.38

        # Save PNG button
        ax_btn = self.fig.add_axes([0.78, max(btn_y, 0.07), 0.20, 0.06])
        self._btn = Button(ax_btn, "Save PNG")
        self._btn.on_clicked(self._on_save)

    # ── widget callbacks ──────────────────────────────────────────────────────

    def _on_source(self, label):
        self.current_source_b = label
        self.update()

    def _on_threshold(self, label):
        thr = float(label)
        if thr in self.active_thrs:
            self.active_thrs.discard(thr)
        else:
            self.active_thrs.add(thr)
        self.update()

    def _on_save(self, _event):
        out_dir = settings.FIGURES_DIR / "compare_pitch"
        out_dir.mkdir(parents=True, exist_ok=True)
        thr_tag = ("_thr" + "_".join(f"{t:.2f}" for t in sorted(self.active_thrs))
                   if self.active_thrs else "")
        fname = (f"{self.recording_id}_{self.extractor_a}_vs_{self.extractor_b}"
                 f"_{self.current_source_b}{thr_tag}.png")
        self.fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        print(f"→ {out_dir / fname}")

    # ── render ────────────────────────────────────────────────────────────────

    def update(self):
        time_b, f0_b_raw, conf_b = self.data_b[self.current_source_b]

        # Active threshold tracks for B
        thrs_sorted = sorted(self.active_thrs) if self.active_thrs else [None]
        b_tracks = []
        for thr in thrs_sorted:
            f0    = apply_thr(f0_b_raw, conf_b, thr) if thr is not None else f0_b_raw
            cents = hz_to_cents(f0, self.tonic)
            color = THR_COLORS.get(thr, self.color_b) if thr is not None else self.color_b
            label = f"thr={thr:.2f}" if thr is not None else self.label_b
            b_tracks.append((time_b, cents, color, label))

        # Align every B track to A's grid (all share the same time_b → same result)
        aligned_b = []
        for tbx, cents_bx, col, lbl in b_tracks:
            t_al, ca_al, cb_al = align_cents(self.time_a, self.cents_a, tbx, cents_bx)
            pct_v = np.isfinite(cb_al).sum() / max(len(t_al), 1) * 100
            aligned_b.append((t_al, ca_al, cb_al, col, lbl, pct_v))

        multi_thr = len(aligned_b) > 1

        # Primary track (lowest / first active threshold) — used for diff plots
        time, cents_a, cents_b0, color_b0, lbl_b0, pct_voiced_b0 = aligned_b[0]

        voiced_a    = np.isfinite(cents_a)
        voiced_b0   = np.isfinite(cents_b0)
        both_voiced = voiced_a & voiced_b0
        diff        = np.where(both_voiced, cents_b0 - cents_a, np.nan)
        abs_diff    = np.where(both_voiced, np.abs(diff), np.nan)
        agree_mask  = both_voiced & (abs_diff < AGREE_THR_CENTS)
        disag_mask  = both_voiced & ~agree_mask
        only_a      = voiced_a & ~voiced_b0
        only_b0     = ~voiced_a & voiced_b0

        total      = len(time)
        mae        = float(np.nanmean(np.abs(diff))) if np.any(np.isfinite(diff)) else 0.0
        bias       = float(np.nanmean(diff))         if np.any(np.isfinite(diff)) else 0.0
        pct_both   = both_voiced.sum() / total * 100
        pct_a_only = only_a.sum() / total * 100
        pct_b_only = only_b0.sum() / total * 100

        src_lbl   = SOURCE_LABELS.get(self.current_source_b, self.current_source_b)
        src_a_lbl = SOURCE_LABELS.get(self.source_a, self.source_a)

        for ax in (self.ax0, self.ax1, self.ax2, self.ax_hist):
            ax.cla()

        # ── ax0: pitch curves ─────────────────────────────────────────────────
        if multi_thr:
            # Per-threshold agree/disagree vs A (each threshold compared independently)
            per_thr = []
            for t_al, ca_al, cb_al, col, lbl, pct_v in aligned_b:
                bv      = np.isfinite(cb_al)
                both_x  = voiced_a & bv
                diff_x  = np.where(both_x, np.abs(cb_al - cents_a), np.nan)
                agree_x = both_x & (diff_x < AGREE_THR_CENTS)
                disag_x = both_x & ~agree_x
                only_bx = bv & ~voiced_a
                per_thr.append((cb_al, agree_x, disag_x, only_bx, bv, col, lbl, pct_v))

            any_b_agrees = np.zeros(total, dtype=bool)
            any_b_voiced = np.zeros(total, dtype=bool)
            for _, agree_x, _, _, bv, *_ in per_thr:
                any_b_agrees |= agree_x
                any_b_voiced |= bv

            a_grey      = voiced_a & any_b_agrees
            a_own_color = voiced_a & any_b_voiced & ~any_b_agrees
            a_only      = voiced_a & ~any_b_voiced

            # Grey agree lines go to the background (z=2) so they never cover diverge colors
            _masked_line(self.ax0, time, cents_a, a_grey, AGREE_COLOR, lw=0.9, zorder=2)
            for cb_x, agree_x, disag_x, only_bx, bv, col, lbl, pct_v in per_thr:
                _masked_line(self.ax0, time, cb_x, agree_x, AGREE_COLOR, lw=0.8, zorder=2)

            # Diverge lines: most permissive (low thr) first → lowest z; least permissive on top
            # per_thr is in ascending threshold order (thrs_sorted)
            n_thr = len(per_thr)
            for i, (cb_x, agree_x, disag_x, only_bx, bv, col, lbl, pct_v) in enumerate(per_thr):
                z = 3 + i
                _masked_line(self.ax0, time, cb_x, disag_x, col, lw=0.9, zorder=z)
                _masked_line(self.ax0, time, cb_x, only_bx, col, lw=0.7, zorder=z, alpha=0.7)

            # A reference on top of everything
            _masked_line(self.ax0, time, cents_a, a_own_color, DISAG_A_COLOR, lw=0.9, zorder=3+n_thr)
            _masked_line(self.ax0, time, cents_a, a_only,      ONLY_A_COLOR,  lw=0.7, zorder=3+n_thr)

            a_pct = voiced_a.sum() / total * 100
            handles = [
                mpatches.Patch(color=AGREE_COLOR,
                               label=f"agree <{AGREE_THR_CENTS:.0f}¢  ({any_b_agrees.sum():,} fr)"),
                Line2D([0], [0], color=DISAG_A_COLOR, lw=1.0,
                       label=f"{self.label_a} diverges from all  ({a_own_color.sum():,} fr)"),
                Line2D([0], [0], color=ONLY_A_COLOR, lw=1.0,
                       label=f"only {self.label_a}  ({a_only.sum()/total*100:.1f}%)"),
            ]
            for _, agree_x, disag_x, only_bx, bv, col, lbl, pct_v in per_thr:
                handles.append(Line2D([0], [0], color=col, lw=1.5,
                                      label=f"{self.label_b} {lbl}  "
                                            f"({pct_v:.0f}% voiced · {disag_x.sum():,} fr diverge)"))
            self.ax0.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.85)
            agree_pct = any_b_agrees.sum() / total * 100
            title_stats = (
                f"tonic={self.tonic:.1f} Hz   A voiced={a_pct:.1f}%   "
                f"agree={agree_pct:.1f}%   "
                f"primary ({lbl_b0}): MAE={mae:.1f}¢  bias={bias:+.1f}¢"
            )
        else:
            # Single-threshold: agree/disagree color coding
            _masked_line(self.ax0, time, cents_a,  agree_mask, AGREE_COLOR,   lw=0.9, zorder=4)
            _masked_line(self.ax0, time, cents_b0, agree_mask, AGREE_COLOR,   lw=0.9, zorder=4)
            _masked_line(self.ax0, time, cents_a,  disag_mask, DISAG_A_COLOR, lw=0.8, zorder=3)
            _masked_line(self.ax0, time, cents_b0, disag_mask, DISAG_B_COLOR, lw=0.8, zorder=3)
            _masked_line(self.ax0, time, cents_a,  only_a,     ONLY_A_COLOR,  lw=0.7, zorder=3)
            _masked_line(self.ax0, time, cents_b0, only_b0,    color_b0,      lw=0.7, zorder=3)
            handles = [
                mpatches.Patch(color=AGREE_COLOR,
                               label=f"agree <{AGREE_THR_CENTS:.0f}¢  "
                                     f"({agree_mask.sum():,} fr, {agree_mask.sum()/total*100:.1f}%)"),
                mpatches.Patch(color=DISAG_A_COLOR,
                               label=f"{self.label_a} diverges  ({disag_mask.sum():,} fr)"),
                mpatches.Patch(color=DISAG_B_COLOR,
                               label=f"{self.label_b} diverges  ({disag_mask.sum():,} fr)"),
                mpatches.Patch(color=ONLY_A_COLOR,
                               label=f"only {self.label_a}  ({pct_a_only:.1f}%)"),
                mpatches.Patch(color=color_b0,
                               label=f"only {self.label_b} [{lbl_b0}]  ({pct_b_only:.1f}%)"),
            ]
            self.ax0.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.85)
            agree_pct = agree_mask.sum() / total * 100
            title_stats = (
                f"tonic={self.tonic:.1f} Hz   both:{pct_both:.1f}%   "
                f"agree={agree_pct:.1f}%   MAE={mae:.1f}¢   bias={bias:+.1f}¢"
            )

        self.ax0.set_ylabel("Cents re tonic")
        self.ax0.set_title(
            f"{self.recording_id}  |  {self.label_a} [{src_a_lbl}]  vs  "
            f"{self.label_b} [{src_lbl}]\n{title_stats}",
            fontsize=9,
        )

        # ── ax1: |Δ| cents ────────────────────────────────────────────────────
        if multi_thr:
            for t_al, ca_al, cb_al, col, lbl, _ in aligned_b:
                n = len(time)
                d_x = np.where(np.isfinite(cb_al[:n]) & np.isfinite(ca_al[:n]),
                               cb_al[:n] - ca_al[:n], np.nan)
                mae_x = float(np.nanmean(np.abs(d_x))) if np.any(np.isfinite(d_x)) else 0.0
                self.ax1.plot(time[:n], np.abs(d_x), color=col, lw=0.75, alpha=0.8,
                              label=f"{lbl}  MAE={mae_x:.1f}¢")
            self.ax1.legend(loc="upper right", fontsize=7, framealpha=0.85)
        else:
            self.ax1.fill_between(time, 0, np.abs(diff),
                                  where=np.isfinite(diff), color="orchid", alpha=0.7)
            self.ax1.axhline(mae, color="purple", lw=1, ls="--", label=f"MAE={mae:.1f}¢")
            self.ax1.legend(loc="upper right", fontsize=8)
        self.ax1.set_ylabel("|Δ| cents")

        # ── ax2: signed Δ (B − A) ─────────────────────────────────────────────
        if multi_thr:
            self.ax2.axhline(0, color="black", lw=0.8, zorder=5)
            for t_al, ca_al, cb_al, col, lbl, _ in aligned_b:
                n = len(time)
                d_x = np.where(np.isfinite(cb_al[:n]) & np.isfinite(ca_al[:n]),
                               cb_al[:n] - ca_al[:n], np.nan)
                bias_x = float(np.nanmean(d_x)) if np.any(np.isfinite(d_x)) else 0.0
                self.ax2.plot(time[:n], d_x, color=col, lw=0.75, alpha=0.75,
                              label=f"{lbl}  bias={bias_x:+.1f}¢")
            self.ax2.legend(loc="upper right", fontsize=7, framealpha=0.85)
        else:
            self.ax2.fill_between(time, 0, diff,
                                  where=np.isfinite(diff) & (diff > 0),
                                  color=DISAG_A_COLOR, alpha=0.6)
            self.ax2.fill_between(time, 0, diff,
                                  where=np.isfinite(diff) & (diff < 0),
                                  color=DISAG_B_COLOR, alpha=0.6)
            self.ax2.axhline(0,    color="black",  lw=0.8)
            self.ax2.axhline(bias, color="purple", lw=1, ls="--", label=f"bias={bias:+.1f}¢")
            self.ax2.legend(loc="upper right", fontsize=8)
        self.ax2.set_ylabel("Δ cents (B − A)")
        self.ax2.set_xlabel("Time (s)")

        # ── divergence region highlights ──────────────────────────────────────
        regions = _top_divergence_regions(time, diff, TOP_N_REGIONS, REGION_SEC)
        if regions:
            cols_r = plt.cm.autumn(np.linspace(0.2, 0.8, len(regions)))
            for ax in (self.ax0, self.ax1, self.ax2):
                for (s, e, _), col in zip(regions, cols_r):
                    ax.axvspan(s, e, color=col, alpha=0.07)

        # ── ax_hist ───────────────────────────────────────────────────────────
        if multi_thr:
            # Summary bar chart: voiced% and MAE per threshold
            thr_labels = [lbl for _, _, _, _, lbl, _ in aligned_b]
            thr_colors = [col for _, _, _, col, _, _ in aligned_b]
            pct_voiced = [pv  for _, _, _, _, _, pv  in aligned_b]
            mae_vals   = []
            for t_al, ca_al, cb_al, col, lbl, _ in aligned_b:
                n   = len(time)
                d_x = np.where(np.isfinite(cb_al[:n]) & np.isfinite(ca_al[:n]),
                               cb_al[:n] - ca_al[:n], np.nan)
                mae_vals.append(float(np.nanmean(np.abs(d_x)))
                                if np.any(np.isfinite(d_x)) else 0.0)

            y_pos = np.arange(len(thr_labels))
            bars  = self.ax_hist.barh(y_pos, pct_voiced,
                                      color=thr_colors, alpha=0.8, height=0.6)
            self.ax_hist.set_yticks(y_pos)
            self.ax_hist.set_yticklabels(thr_labels, fontsize=8)
            self.ax_hist.set_xlabel("Voiced %", fontsize=8)
            for bar, mae_v in zip(bars, mae_vals):
                self.ax_hist.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                                  f"MAE={mae_v:.1f}¢", va="center", fontsize=7)
            a_pct = voiced_a.sum() / total * 100
            self.ax_hist.axvline(a_pct, color=self.color_a, lw=1.5, ls="--")
            self.ax_hist.text(a_pct + 0.3, len(thr_labels) - 0.5,
                              f"{self.label_a}\n{a_pct:.1f}%",
                              color=self.color_a, fontsize=7, va="top")
            self.ax_hist.set_title(f"{self.label_b}\nVoicing vs threshold", fontsize=9)
        else:
            bins     = np.arange(-1200, 2401, 50)
            a_only_c = cents_a[only_a]
            b_only_c = cents_b0[only_b0]
            if len(a_only_c):
                self.ax_hist.barh(bins[:-1], np.histogram(a_only_c, bins=bins)[0],
                                  height=48, color=ONLY_A_COLOR, alpha=0.8,
                                  label=f"only {self.label_a}")
            if len(b_only_c):
                self.ax_hist.barh(bins[:-1], -np.histogram(b_only_c, bins=bins)[0],
                                  height=48, color=color_b0, alpha=0.8,
                                  label=f"only {self.label_b}")
            self.ax_hist.axhline(0, color="black", lw=0.5)
            self.ax_hist.axvline(0, color="black", lw=0.8)
            self.ax_hist.set_xlabel(f"← {self.label_b}   {self.label_a} →")
            self.ax_hist.set_ylabel("Cents re tonic")
            self.ax_hist.legend(fontsize=7)
            self.ax_hist.set_title("Exclusive voiced\nframes", fontsize=9)

        self.fig.canvas.draw_idle()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Interactive pitch comparison on CV corpus recordings."
    )
    parser.add_argument("recordings", nargs="*",
                        help="Recording IDs (default: all RECORDING_SELECTION)")
    parser.add_argument("--all", action="store_true",
                        help="Process all recordings in settings.RECORDING_SELECTION")
    parser.add_argument("--source",   default="original",
                        help="Source for extractor A (default: original)")
    parser.add_argument("--extractor-a", default="ftanet",
                        choices=list(EXTRACTOR_CV_DIR.keys()))
    parser.add_argument("--extractor-b", default="swiftf0scratch",
                        choices=list(EXTRACTOR_CV_DIR.keys()))
    args = parser.parse_args()

    recordings = (settings.RECORDING_SELECTION
                  if (args.all or not args.recordings)
                  else args.recordings)

    plt.ion()  # non-blocking: all figures open simultaneously

    figs = []
    for rec_id in recordings:
        try:
            fig = InteractiveFig(rec_id, args.extractor_a, args.extractor_b, args.source)
            figs.append(fig)
        except FileNotFoundError as e:
            print(f"[skip] {rec_id}: {e}")

    if not figs:
        print("No valid recordings found.")
        return

    print(f"\nOpened {len(figs)} figure(s). Close windows or Ctrl-C to quit.")
    plt.show(block=True)


if __name__ == "__main__":
    main()
