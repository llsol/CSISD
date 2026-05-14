"""
Extract CP pitch deviation curves from the GT corpus.

For each CP segment we store:
    - curve    : np.ndarray (L_CANONICAL,) — pitch deviation in cents
                  (raw cents − median of segment, resampled to L_CANONICAL)
    - mean_cents : float — absolute pitch level (cents rel. tonic)
    - dur_sec    : float
    - recording_id, svara_label

The model learns micro-variations (ondulations, slopes, tuning drift)
on top of the absolute pitch level, which is provided by the GRU+VAE.

Usage:
    python -m src.models.curve_vae.extract_cp_curves
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.interpolate import interp1d

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
)

L_CANONICAL = 64    # fixed output length after resampling
MIN_SAMPLES = 6     # discard CP segments shorter than this

DEFAULT_OUT = S.CURVE_VAE_DIR / "gt_cp_curves.parquet"


# ---------------------------------------------------------------------------

def _resample(arr: np.ndarray, L: int) -> np.ndarray:
    """Resample arr (length N) to length L using linear interpolation."""
    if len(arr) == L:
        return arr.astype(np.float32)
    t_in  = np.linspace(0, 1, len(arr))
    t_out = np.linspace(0, 1, L)
    return interp1d(t_in, arr, kind="linear")(t_out).astype(np.float32)


def extract_one_recording(
    recording_id: str,
    tonic_hz: float,
    min_samples: int = MIN_SAMPLES,
    l_canonical: int = L_CANONICAL,
) -> list[dict]:
    interim_root = S.INTERIM_RECORDINGS
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

    ann_path  = corpus_root / recording_id / "raw" / f"{recording_id}_ann_svara.tsv"
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

        times_svara = df_svara["time_rel_sec"].to_numpy()

        for seg in segments:
            if seg["type"] != "CP":
                continue

            s, e = seg["start"], seg["end"]
            raw_cents = cents_all[df_svara["row_idx"][s:e].to_numpy()]

            valid = np.isfinite(raw_cents)
            if valid.sum() < min_samples:
                continue
            raw_cents = raw_cents[valid]

            t_seg   = times_svara[s:e][valid]
            dur_sec = float(t_seg[-1] - t_seg[0])

            mean_cents = float(np.median(raw_cents))
            deviation  = raw_cents - mean_cents          # micro-variations around median

            resampled = _resample(deviation, l_canonical)

            results.append({
                "uid":          f"{recording_id}_{uid:05d}",
                "recording_id": recording_id,
                "svara_label":  ann["svara_label"],
                "n_samples":    len(raw_cents),
                "dur_sec":      dur_sec,
                "mean_cents":   mean_cents,
                "curve":        resampled.tolist(),
            })
            uid += 1

    return results


def extract_corpus(
    recording_ids: list[str] | None = None,
    tonic_map: dict[str, float] | None = None,
    min_samples: int = MIN_SAMPLES,
    l_canonical: int = L_CANONICAL,
) -> pl.DataFrame:
    if recording_ids is None:
        recording_ids = S.SARASUDA_VARNAM
    if tonic_map is None:
        tonic_map = S.SARASUDA_TONICS

    all_rows = []
    for rid in recording_ids:
        print(f"  {rid}...")
        rows = extract_one_recording(rid, tonic_map[rid], min_samples, l_canonical)
        all_rows.extend(rows)
        print(f"    → {len(rows)} CP segments")

    print(f"Total: {len(all_rows)} CP curves")
    return pl.DataFrame(all_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-samples",  type=int, default=MIN_SAMPLES)
    parser.add_argument("--l-canonical",  type=int, default=L_CANONICAL)
    parser.add_argument("--out",          default=str(DEFAULT_OUT))
    args = parser.parse_args()

    df = extract_corpus(min_samples=args.min_samples, l_canonical=args.l_canonical)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)
    print(f"Saved → {out}")
    print(f"  dur_sec: mean={df['dur_sec'].mean():.3f}  "
          f"p5={df['dur_sec'].quantile(0.05):.3f}  "
          f"p95={df['dur_sec'].quantile(0.95):.3f}")
    dev = np.concatenate([np.array(r) for r in df["curve"].to_list()])
    print(f"  deviation: std={dev.std():.2f}¢  max_abs={np.abs(dev).max():.2f}¢")


if __name__ == "__main__":
    main()
