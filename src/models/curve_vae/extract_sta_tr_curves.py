"""
Extract normalized pitch curves for STA and TR segments from the GT corpus.

For each STA/TR segment we store:
    - t_norm   : np.ndarray (K,) — normalized time [0, 1]
    - p_norm   : np.ndarray (K,) — normalized pitch [0, 1]
                   STA: (cents - cents[0]) / (cents[-1] - cents[0])   0 → 1
                   TR : (cents - cents[0]) / (cents[-1] - cents[0])   0 → 1
    - p_start_cents / p_end_cents : raw cents values at endpoints
    - dur_sec  : segment duration in seconds
    - recording_id, svara_label, seg_type, segment_uid

Usage:
    python -m src.models.curve_vae.extract_curves
    python -m src.models.curve_vae.extract_curves --min-samples 8 --out data/interim/curves.parquet
"""

from __future__ import annotations

import argparse
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

DEFAULT_OUT = S.DATA_INTERIM / "models" / "curve_vae" / "gt_curves.parquet"
MIN_SAMPLES = 6   # discard segments shorter than this


# ---------------------------------------------------------------------------

def extract_one_recording(
    recording_id: str,
    tonic_hz: float,
    min_samples: int = MIN_SAMPLES,
    seg_types: tuple[str, ...] = ("STAp", "STAt", "TRa", "TRd"),
) -> list[dict]:
    """Extract all STA/TR normalized curves for one recording."""
    interim_root = Path("data/interim")
    corpus_root  = Path("data/corpus")

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

    ann_path = corpus_root / recording_id / "raw" / f"{recording_id}_ann_svara.tsv"
    df_svaras = load_annotations(ann_path, annotation_type="svara", engine="polars")

    peak_row_map = map_peaks_to_global_rows(df_pitch, df_peaks)
    t_all        = df_pitch["time_rel_sec"].to_numpy()
    cents_all    = df_pitch["f0_savgol_p3_w13_cents"].to_numpy()

    results = []
    uid = 0

    for ann in df_svaras.iter_rows(named=True):
        t_start = float(ann["start_time_sec"])
        t_end   = float(ann["end_time_sec"])
        if t_end < t_start:
            t_start, t_end = t_end, t_start

        mask     = (t_all >= t_start) & (t_all <= t_end)
        df_svara = df_pitch.filter(pl.Series(mask))
        if df_svara.is_empty():
            continue

        df_svara     = label_samples_sil_cp_sta(df_svara)
        local_pk_map = restrict_peaks_to_slice(df_svara, peak_row_map)
        segments     = build_segments_for_one_svara(df_svara, local_pk_map)
        segments     = assign_segment_cents(segments, df_svara, local_pk_map)

        times_svara = df_svara["time_rel_sec"].to_numpy()

        for seg in segments:
            if seg["type"] not in seg_types:
                continue

            s, e = seg["start"], seg["end"]
            raw_cents = cents_all[
                df_svara["row_idx"][s : e].to_numpy()
            ]

            # drop NaN / inf
            valid = np.isfinite(raw_cents)
            if valid.sum() < min_samples:
                continue
            raw_cents = raw_cents[valid]
            K = len(raw_cents)

            t_seg   = times_svara[s : e][valid]
            dur_sec = float(t_seg[-1] - t_seg[0]) if K > 1 else 0.0
            t_norm  = (t_seg - t_seg[0]) / dur_sec if dur_sec > 0 else np.linspace(0, 1, K)

            p_start = float(raw_cents[0])
            p_end   = float(raw_cents[-1])
            delta   = p_end - p_start

            if abs(delta) < 1e-6:
                continue
            p_norm = (raw_cents - p_start) / delta   # 0 → 1 for both STA and TR

            results.append({
                "uid":            f"{recording_id}_{uid:05d}",
                "recording_id":   recording_id,
                "svara_label":    ann["svara_label"],
                "seg_type":       seg["type"],
                "n_samples":      K,
                "dur_sec":        dur_sec,
                "p_start_cents":  p_start,
                "p_end_cents":    p_end,
                "t_norm":         t_norm.tolist(),
                "p_norm":         p_norm.tolist(),
            })
            uid += 1

    return results


def extract_corpus(
    recording_ids: list[str] | None = None,
    tonic_map: dict[str, float] | None = None,
    min_samples: int = MIN_SAMPLES,
    seg_types: tuple[str, ...] = ("STAp", "STAt", "TRa", "TRd"),
) -> pl.DataFrame:
    if recording_ids is None:
        recording_ids = S.SARASUDA_VARNAM
    if tonic_map is None:
        tonic_map = S.SARASUDA_TONICS

    all_rows = []
    for rid in recording_ids:
        print(f"  {rid}...")
        rows = extract_one_recording(rid, tonic_map[rid], min_samples, seg_types)
        all_rows.extend(rows)
        print(f"    → {len(rows)} segments")

    print(f"Total: {len(all_rows)} curves")
    return pl.DataFrame(all_rows)


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract GT pitch curves for STA/TR segments.")
    parser.add_argument("--min-samples", type=int, default=MIN_SAMPLES)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    df = extract_corpus(min_samples=args.min_samples)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)
    print(f"Saved → {out}")

    for seg_type in ("STAp", "STAt", "TRa", "TRd"):
        sub = df.filter(pl.col("seg_type") == seg_type)
        if len(sub) == 0:
            continue
        print(f"  {seg_type}: {len(sub)} curves  "
              f"dur={sub['dur_sec'].mean():.3f}s mean  "
              f"samples={sub['n_samples'].mean():.1f} mean")


if __name__ == "__main__":
    main()
