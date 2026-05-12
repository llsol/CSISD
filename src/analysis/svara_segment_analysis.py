"""
Svara segment analysis — data loading and feature extraction.

For each annotated svara, extracts CP/STA/TR/SIL segment statistics
(durations, cents, run-level features) and returns them as a list of dicts
and a Polars DataFrame.

Usage (standalone — loads all 5 Sarasuda recordings):
    python -m src.analysis.svara_segment_analysis
    python -m src.analysis.svara_segment_analysis --recordings srs_v1_bdn_sav srs_v1_drn_sav

Importable API:
    from src.analysis.svara_segment_analysis import load_all, PERFORMER, COLORS

    all_rows, df, svara_labels, performers = load_all()
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import settings as S
from src.io.pitch_io import load_preprocessed_pitch, load_flat_regions, load_peaks
from src.io.annotation_io import load_annotations
from src.features.structural_embedding import (
    label_samples_sil_cp_sta,
    map_peaks_to_global_rows,
    restrict_peaks_to_slice,
    build_segments_for_one_svara,
    assign_segment_cents,
)

# ── constants ─────────────────────────────────────────────────────────────────

RECORDINGS = S.SARASUDA_VARNAM
PERFORMER  = {rec: rec.split('_')[2] for rec in RECORDINGS}
COLORS     = {
    'bdn': '#e41a1c', 'drn': '#377eb8', 'psn': '#4daf4a',
    'rkm': '#984ea3', 'svd': '#ff7f00',
}
SCALE_ORDER = ['S', 'R', 'G', 'M', 'P', 'D', 'N']


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_recording(recording_id: str):
    """Load pitch, peaks and svara annotations for one recording."""
    tonic_hz = S.SARASUDA_TONICS[recording_id]

    df_pitch = load_preprocessed_pitch(
        recording_id=recording_id,
        root_dir=S.DATA_INTERIM,
        tonic_hz=tonic_hz,
        convert_to_cents=True,
    )
    df_flat  = load_flat_regions(recording_id=recording_id, root_dir=S.DATA_INTERIM)
    df_peaks = load_peaks(recording_id=recording_id, root_dir=S.DATA_INTERIM)

    df_pitch = (
        df_pitch
        .join(df_flat.select(['time_rel_sec', 'flat_region']), on='time_rel_sec', how='left')
        .with_columns(pl.col('flat_region').fill_null(False))
        .with_row_index('row_idx')
    )

    ann_path  = S.DATA_CORPUS / recording_id / 'raw' / f'{recording_id}_ann_svara.tsv'
    df_svaras = load_annotations(file_path=ann_path, annotation_type='svara', engine='polars')

    return df_pitch, df_peaks, df_svaras


# ── feature extraction ────────────────────────────────────────────────────────

def analyse_one_recording(
    recording_id: str,
    df_pitch,
    df_peaks,
    df_svaras,
) -> list[dict]:
    """
    Per each annotated svara: extract CP/STAp/STAt/TRa/TRd/SIL segments and compute
    duration, cents, and run-level features.

    Returns a list of dicts — one per svara occurrence.
    Scalar fields are safe for pl.DataFrame; list fields are kept for plots.
    """
    peak_row_map = map_peaks_to_global_rows(df_pitch, df_peaks)
    t_all        = df_pitch['time_rel_sec'].to_numpy()
    performer    = PERFORMER[recording_id]
    rows         = []

    for ann in df_svaras.iter_rows(named=True):
        t_start     = float(ann['start_time_sec'])
        t_end       = float(ann['end_time_sec'])
        svara_label = ann['svara_label']

        mask     = (t_all >= t_start) & (t_all <= t_end)
        df_svara = df_pitch.filter(pl.Series(mask))
        if df_svara.is_empty():
            continue

        df_svara       = label_samples_sil_cp_sta(df_svara)
        local_peak_map = restrict_peaks_to_slice(df_svara, peak_row_map)
        segments       = build_segments_for_one_svara(df_svara=df_svara, local_peak_map=local_peak_map)
        segments       = assign_segment_cents(
            segments=segments, df_svara=df_svara, local_peak_map=local_peak_map,
        )

        t_svara   = df_svara['time_rel_sec'].to_numpy()
        dt        = float(np.median(np.diff(t_svara))) if len(t_svara) > 1 else 0.0
        svara_dur = float(t_end - t_start)

        cp_segs   = [s for s in segments if s['type'] == 'CP']
        stap_segs = [s for s in segments if s['type'] == 'STAp']
        stat_segs = [s for s in segments if s['type'] == 'STAt']
        sta_segs  = stap_segs + stat_segs   # aggregate for backward compat
        tra_segs  = [s for s in segments if s['type'] == 'TRa']
        trd_segs  = [s for s in segments if s['type'] == 'TRd']
        tr_segs   = tra_segs + trd_segs     # aggregate
        sil_segs  = [s for s in segments if s['type'] == 'SIL']

        # [A] durations
        cp_dur_list   = [(s['end'] - s['start']) * dt for s in cp_segs]
        stap_dur_list = [(s['end'] - s['start']) * dt for s in stap_segs]
        stat_dur_list = [(s['end'] - s['start']) * dt for s in stat_segs]
        sta_dur_list  = stap_dur_list + stat_dur_list
        tra_dur_list  = [(s['end'] - s['start']) * dt for s in tra_segs]
        trd_dur_list  = [(s['end'] - s['start']) * dt for s in trd_segs]
        tr_dur_list   = tra_dur_list + trd_dur_list
        cp_dur   = sum(cp_dur_list)
        stap_dur = sum(stap_dur_list)
        stat_dur = sum(stat_dur_list)
        sta_dur  = stap_dur + stat_dur
        tra_dur  = sum(tra_dur_list)
        trd_dur  = sum(trd_dur_list)
        tr_dur   = tra_dur + trd_dur
        sil_dur  = sum((s['end'] - s['start']) * dt for s in sil_segs)

        # [D] cents
        def _cents(segs):
            return [s['cents'] for s in segs
                    if s.get('cents') is not None and np.isfinite(s.get('cents', np.nan))]

        cp_cents   = _cents(cp_segs)
        stap_cents = _cents(stap_segs)   # STAp = peaks
        stat_cents = _cents(stat_segs)   # STAt = troughs
        sta_cents  = stap_cents + stat_cents

        # [C] run-level
        def _runs(*seg_types):
            run_durs, cur = [], 0.0
            for s in segments:
                if s['type'] in seg_types:
                    cur += (s['end'] - s['start']) * dt
                elif cur > 0:
                    run_durs.append(cur); cur = 0.0
            if cur > 0:
                run_durs.append(cur)
            return run_durs

        cp_run_durs   = _runs('CP')
        stap_run_durs = _runs('STAp')
        stat_run_durs = _runs('STAt')
        sta_run_durs  = _runs('STAp', 'STAt')
        tra_run_durs  = _runs('TRa')
        trd_run_durs  = _runs('TRd')
        tr_run_durs   = _runs('TRa', 'TRd')
        sta_tr_run_durs = _runs('STAp', 'STAt', 'TRa', 'TRd')

        rows.append({
            'recording_id':      recording_id,
            'performer':         performer,
            'svara_label':       svara_label,
            'svara_dur_sec':     svara_dur,
            # [A] segment counts + totals (aggregate)
            'n_cp':              len(cp_segs),
            'cp_total_dur_sec':  float(cp_dur),
            'cp_frac':           float(cp_dur  / max(svara_dur, 1e-6)),
            'n_sta':             len(sta_segs),
            'sta_total_dur_sec': float(sta_dur),
            'sta_frac':          float(sta_dur / max(svara_dur, 1e-6)),
            'n_tr':              len(tr_segs),
            'tr_total_dur_sec':  float(tr_dur),
            'tr_frac':           float(tr_dur  / max(svara_dur, 1e-6)),
            'sil_total_dur_sec': float(sil_dur),
            'sil_frac':          float(sil_dur / max(svara_dur, 1e-6)),
            # [A] split STAp / STAt
            'n_stap':              len(stap_segs),
            'stap_total_dur_sec':  float(stap_dur),
            'stap_frac':           float(stap_dur / max(svara_dur, 1e-6)),
            'n_stat':              len(stat_segs),
            'stat_total_dur_sec':  float(stat_dur),
            'stat_frac':           float(stat_dur / max(svara_dur, 1e-6)),
            # [A] split TRa / TRd
            'n_tra':              len(tra_segs),
            'tra_total_dur_sec':  float(tra_dur),
            'tra_frac':           float(tra_dur / max(svara_dur, 1e-6)),
            'n_trd':              len(trd_segs),
            'trd_total_dur_sec':  float(trd_dur),
            'trd_frac':           float(trd_dur / max(svara_dur, 1e-6)),
            # [D] cents summaries
            'cp_mean_cents':          float(np.mean(cp_cents))    if cp_cents    else np.nan,
            'sta_mean_cents':         float(np.mean(sta_cents))   if sta_cents   else np.nan,
            'sta_std_cents':          float(np.std(sta_cents))    if sta_cents   else np.nan,
            'stap_mean_cents':        float(np.mean(stap_cents))  if stap_cents  else np.nan,
            'stap_std_cents':         float(np.std(stap_cents))   if stap_cents  else np.nan,
            'stat_mean_cents':        float(np.mean(stat_cents))  if stat_cents  else np.nan,
            'stat_std_cents':         float(np.std(stat_cents))   if stat_cents  else np.nan,
            # kept for backward compat — same as stap/stat_mean_cents
            'sta_peaks_mean_cents':   float(np.mean(stap_cents))  if stap_cents  else np.nan,
            'sta_peaks_std_cents':    float(np.std(stap_cents))   if stap_cents  else np.nan,
            'sta_valleys_mean_cents': float(np.mean(stat_cents))  if stat_cents  else np.nan,
            'sta_valleys_std_cents':  float(np.std(stat_cents))   if stat_cents  else np.nan,
            # [C] run counts + summaries (aggregate)
            'n_cp_runs':        len(cp_run_durs),
            'n_sta_runs':       len(sta_run_durs),
            'n_tr_runs':        len(tr_run_durs),
            'cp_run_mean_dur':  float(np.mean(cp_run_durs))   if cp_run_durs   else np.nan,
            'sta_run_mean_dur': float(np.mean(sta_run_durs))  if sta_run_durs  else np.nan,
            'tr_run_mean_dur':  float(np.mean(tr_run_durs))   if tr_run_durs   else np.nan,
            'cp_run_max_dur':      float(np.max(cp_run_durs))      if cp_run_durs      else np.nan,
            'cp_run_max_frac':     float(np.max(cp_run_durs) / max(svara_dur, 1e-6))   if cp_run_durs      else np.nan,
            'sta_run_max_dur':     float(np.max(sta_run_durs))     if sta_run_durs     else np.nan,
            'sta_tr_run_max_dur':  float(np.max(sta_tr_run_durs))  if sta_tr_run_durs  else np.nan,
            'sta_tr_run_max_frac': float(np.max(sta_tr_run_durs) / max(svara_dur, 1e-6)) if sta_tr_run_durs else np.nan,
            # [C] split run summaries
            'n_stap_runs':    len(stap_run_durs),
            'n_stat_runs':    len(stat_run_durs),
            'n_tra_runs':     len(tra_run_durs),
            'n_trd_runs':     len(trd_run_durs),
            'stap_run_mean_dur': float(np.mean(stap_run_durs)) if stap_run_durs else np.nan,
            'stat_run_mean_dur': float(np.mean(stat_run_durs)) if stat_run_durs else np.nan,
            'tra_run_mean_dur':  float(np.mean(tra_run_durs))  if tra_run_durs  else np.nan,
            'trd_run_mean_dur':  float(np.mean(trd_run_durs))  if trd_run_durs  else np.nan,
            # lists — excluded from DataFrame, kept for plots
            'cp_dur_list':          cp_dur_list,
            'stap_dur_list':        stap_dur_list,
            'stat_dur_list':        stat_dur_list,
            'sta_dur_list':         sta_dur_list,
            'tra_dur_list':         tra_dur_list,
            'trd_dur_list':         trd_dur_list,
            'tr_dur_list':          tr_dur_list,
            'cp_cents_list':        cp_cents,
            'stap_cents_list':      stap_cents,
            'stat_cents_list':      stat_cents,
            'sta_cents_list':       sta_cents,
            'cp_run_dur_list':      cp_run_durs,
            'stap_run_dur_list':    stap_run_durs,
            'stat_run_dur_list':    stat_run_durs,
            'sta_run_dur_list':     sta_run_durs,
            'tra_run_dur_list':     tra_run_durs,
            'trd_run_dur_list':     trd_run_durs,
            'tr_run_dur_list':      tr_run_durs,
        })

    return rows


# ── public API ────────────────────────────────────────────────────────────────

def load_all(
    recording_ids: list[str] | None = None,
) -> tuple[list[dict], pl.DataFrame, list[str], list[str]]:
    """
    Load and analyse all recordings.

    Returns:
        all_rows     — list of dicts (one per svara, includes list fields)
        df           — Polars DataFrame (scalar fields only)
        svara_labels — svara labels present, in scale order
        performers   — sorted performer codes
    """
    recording_ids = recording_ids or RECORDINGS
    all_rows: list[dict] = []
    for rec in recording_ids:
        print(f'  Loading {rec}...', flush=True)
        df_pitch, df_peaks, df_svaras = load_recording(rec)
        rows = analyse_one_recording(rec, df_pitch, df_peaks, df_svaras)
        all_rows.extend(rows)
        print(f'    → {len(rows)} svaras')

    df = pl.DataFrame([
        {k: v for k, v in r.items() if not isinstance(v, list)}
        for r in all_rows
    ])

    _present     = set(df['svara_label'].unique().to_list())
    svara_labels = [s for s in SCALE_ORDER if s in _present]
    performers   = sorted(df['performer'].unique().to_list())

    return all_rows, df, svara_labels, performers


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Svara segment analysis — load and print summary")
    parser.add_argument("--recordings", nargs="+", default=None,
                        help="Recording IDs to process (default: all Sarasuda)")
    args = parser.parse_args()

    print("[svara_segment_analysis] Loading data...")
    all_rows, df, svara_labels, performers = load_all(args.recordings)

    print(f"\nTotal svaras: {len(all_rows)}")
    print(f"Svara labels: {svara_labels}")
    print(f"Performers:   {performers}")
    print(f"DataFrame:    {df.shape[0]} rows × {df.shape[1]} cols")
    print(df.describe())


if __name__ == "__main__":
    main()
