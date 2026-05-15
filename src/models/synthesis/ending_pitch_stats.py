"""
Statistical study of GT segment pitch boundaries, used by synthesize.py.

Two datasets are extracted and saved:

1. ending_pitch_stats.parquet  — last actual pitch sample of the last non-SIL
   segment per svara.  Used when the last generated segment is TR with no
   following segment.

2. tr_sil_delta_stats.parquet  — for every GT TR segment immediately followed
   by SIL: delta = last_cents − first_cents of the TR (signed, usually < 0).
   Used when a generated TR is followed by SIL: TR.end = TR.start + delta.

Usage
-----
    python -m src.models.synthesis.ending_pitch_stats
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[3]
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

SYNTH_DIR        = S.SYNTHESIS_DIR
OUT_ENDING       = SYNTH_DIR / "ending_pitch_stats.parquet"
OUT_TR_SIL_DELTA = SYNTH_DIR / "tr_sil_delta_stats.parquet"

PITCH_COL = "f0_savgol_p3_w13_cents"


def _extract_one_recording(
    recording_id: str,
    tonic_hz: float,
) -> tuple[list[dict], list[dict]]:
    """
    Returns
    -------
    end_records      : [{recording_id, svara_label, seg_type, last_cents}, ...]
    tr_sil_records   : [{recording_id, svara_label, start_cents, end_cents, delta_cents}, ...]
    """
    corpus_root  = S.DATA_CORPUS
    interim_root = S.INTERIM_RECORDINGS

    df_pitch = load_preprocessed_pitch(
        recording_id=recording_id,
        root_dir=interim_root,
        tonic_hz=tonic_hz,
        convert_to_cents=True,
    )
    df_flat  = load_flat_regions(recording_id=recording_id, root_dir=interim_root)
    df_peaks = load_peaks(recording_id=recording_id, root_dir=interim_root)

    df_pitch = (
        df_pitch
        .join(df_flat.select(["time_rel_sec", "flat_region"]), on="time_rel_sec", how="left")
        .with_columns(pl.col("flat_region").fill_null(False))
        .with_row_index("row_idx")
    )

    ann_path  = corpus_root / recording_id / "raw" / f"{recording_id}_ann_svara.tsv"
    df_svaras = load_annotations(ann_path, annotation_type="svara", engine="polars")

    peak_row_map = map_peaks_to_global_rows(df_pitch, df_peaks)
    t_all        = df_pitch["time_rel_sec"].to_numpy()

    end_records:    list[dict] = []
    tr_sil_records: list[dict] = []

    for ann in df_svaras.iter_rows(named=True):
        t_start = float(ann["start_time_sec"])
        t_end   = float(ann["end_time_sec"])
        if t_end < t_start:
            t_start, t_end = t_end, t_start

        mask     = (t_all >= t_start) & (t_all <= t_end)
        df_svara = df_pitch.filter(pl.Series(mask))
        if df_svara.is_empty():
            continue

        df_svara       = label_samples_sil_cp_sta(df_svara)
        local_peak_map = restrict_peaks_to_slice(df_svara, peak_row_map)
        segments       = build_segments_for_one_svara(df_svara, local_peak_map)
        segments       = assign_segment_cents(segments, df_svara, local_peak_map)

        pitch_arr = df_svara[PITCH_COL].to_numpy()

        # ── 1. last non-SIL ending pitch ────────────────────────────────
        last_seg = next((s for s in reversed(segments) if s["type"] != "SIL"), None)
        if last_seg is not None:
            vals  = pitch_arr[last_seg["start"]: last_seg["end"]]
            valid = vals[np.isfinite(vals)]
            if len(valid) > 0:
                end_records.append({
                    "recording_id": recording_id,
                    "svara_label":  ann["svara_label"],
                    "seg_type":     last_seg["type"],
                    "last_cents":   float(valid[-1]),
                })

        # ── 2. TR→SIL deltas ────────────────────────────────────────────
        for i, seg in enumerate(segments[:-1]):
            if seg["type"] not in ("TRa", "TRd"):
                continue
            if segments[i + 1]["type"] != "SIL":
                continue
            vals  = pitch_arr[seg["start"]: seg["end"]]
            valid = vals[np.isfinite(vals)]
            if len(valid) < 2:
                continue
            tr_start = float(valid[0])
            tr_end   = float(valid[-1])
            tr_sil_records.append({
                "recording_id": recording_id,
                "svara_label":  ann["svara_label"],
                "start_cents":  tr_start,
                "end_cents":    tr_end,
                "delta_cents":  tr_end - tr_start,
            })

    return end_records, tr_sil_records


def compute_and_save(
    recording_ids: list[str] = S.RECORDING_SELECTION,
    tonic_map: dict[str, float] = S.RECORDING_SELECTION_TONICS,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    all_end:    list[dict] = []
    all_tr_sil: list[dict] = []

    for rid in recording_ids:
        print(f"  {rid}…")
        end, tr_sil = _extract_one_recording(rid, tonic_map[rid])
        all_end.extend(end)
        all_tr_sil.extend(tr_sil)

    SYNTH_DIR.mkdir(parents=True, exist_ok=True)

    df_end = pl.DataFrame(all_end)
    df_end.write_parquet(OUT_ENDING)
    print(f"\nEnding stats: {len(df_end)} records → {OUT_ENDING}")
    print(
        df_end.group_by(["svara_label", "seg_type"])
        .agg(pl.len().alias("n"),
             pl.col("last_cents").mean().alias("mean_¢"),
             pl.col("last_cents").std().alias("std_¢"))
        .sort(["svara_label", "seg_type"])
    )

    df_tr_sil = pl.DataFrame(all_tr_sil)
    df_tr_sil.write_parquet(OUT_TR_SIL_DELTA)
    print(f"\nTR→SIL delta stats: {len(df_tr_sil)} records → {OUT_TR_SIL_DELTA}")
    if len(df_tr_sil) > 0:
        print(
            df_tr_sil.group_by("svara_label")
            .agg(pl.len().alias("n"),
                 pl.col("delta_cents").mean().alias("mean_delta_¢"),
                 pl.col("delta_cents").std().alias("std_delta_¢"))
            .sort("svara_label")
        )

    return df_end, df_tr_sil


# ── public loaders ────────────────────────────────────────────────────────────

def load_stats(path: Path = OUT_ENDING) -> pl.DataFrame:
    return pl.read_parquet(path)


def load_tr_sil_stats(path: Path = OUT_TR_SIL_DELTA) -> pl.DataFrame:
    return pl.read_parquet(path)


# ── samplers ──────────────────────────────────────────────────────────────────

def _kde_sample(values: np.ndarray, rng: np.random.Generator, bw_frac: float = 0.4) -> float:
    """Simple kernel density sample: pick random observation, add Gaussian noise."""
    mu  = float(rng.choice(values))
    std = float(values.std()) if len(values) > 1 else 30.0
    return float(rng.normal(mu, std * bw_frac))


def sample_ending_cents(
    df_stats:    pl.DataFrame,
    svara_label: str,
    seg_type:    str,
    rng:         np.random.Generator | None = None,
) -> float:
    """Sample ending pitch (cents) for the last segment of a svara."""
    if rng is None:
        rng = np.random.default_rng()
    rows = df_stats.filter(
        (pl.col("svara_label") == svara_label) & (pl.col("seg_type") == seg_type)
    )["last_cents"].to_numpy()
    if len(rows) == 0:
        rows = df_stats.filter(pl.col("seg_type") == seg_type)["last_cents"].to_numpy()
    if len(rows) == 0:
        return 0.0
    return _kde_sample(rows, rng)


def sample_tr_sil_delta(
    df_stats:    pl.DataFrame,
    svara_label: str,
    rng:         np.random.Generator | None = None,
) -> float:
    """Sample delta_cents for a TR segment followed by SIL (usually negative)."""
    if rng is None:
        rng = np.random.default_rng()
    rows = df_stats.filter(pl.col("svara_label") == svara_label)["delta_cents"].to_numpy()
    if len(rows) == 0:
        rows = df_stats["delta_cents"].to_numpy()
    if len(rows) == 0:
        return -50.0   # fallback: descend half-semitone
    return _kde_sample(rows, rng)


if __name__ == "__main__":
    compute_and_save()
