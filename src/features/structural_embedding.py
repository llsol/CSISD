from __future__ import annotations

from pathlib import Path
import numpy as np
import polars as pl

from settings import SARASUDA_VARNAM
from src.io.pitch_io import load_preprocessed_pitch, load_flat_regions, load_peaks
from src.io.annotation_io import load_annotations


# -------------------------------------------------------------------
# Fixed encoding for segment types
# -------------------------------------------------------------------
TYPE_TO_ONEHOT = {
    "CP":  [1.0, 0.0, 0.0],
    "SIL": [0.0, 1.0, 0.0],
    "STA": [0.0, 0.0, 1.0],
}


# -------------------------------------------------------------------
# PUBLIC API
# -------------------------------------------------------------------
def structural_embedding_one_recording(
    recording_id: str,
    tonic_hz: float,
    corpus_root: Path | str = "data/corpus",
    interim_root: Path | str = "data/interim",
    annotation_path: Path | str | None = None,
    out_path: Path | None = None,
    max_segments: int = 12,
    tau_init_sil: float = 0.30,
) -> pl.DataFrame:
    """
    Build one structural embedding per annotated svara for a single recording.

    Segmentation strategy:
    - svara boundaries come from annotation rows (start_time_sec, end_time_sec)
    - NOT from svara_id inside df_pitch

    Output:
        One row per annotated svara, including:
        - metadata
        - n_segments
        - embedding (list[float])
    """

    corpus_root = Path(corpus_root)
    interim_root = Path(interim_root)

    # ------------------------------------------------------------
    # 1) Load preprocessed pitch and convert to cents
    # ------------------------------------------------------------
    df_pitch = load_preprocessed_pitch(
        recording_id=recording_id,
        root_dir=interim_root,
        tonic_hz=tonic_hz,
        convert_to_cents=True,
    )

    # ------------------------------------------------------------
    # 2) Load flat regions and peaks
    # ------------------------------------------------------------
    df_flat = load_flat_regions(
        recording_id=recording_id,
        root_dir=interim_root,
    )

    df_peaks = load_peaks(
        recording_id=recording_id,
        root_dir=interim_root,
    )

    # Merge flat_region into pitch dataframe
    df_pitch = (
        df_pitch.join(
            df_flat.select(["time_rel_sec", "flat_region"]),
            on="time_rel_sec",
            how="left",
        )
        .with_columns(pl.col("flat_region").fill_null(False))
        .with_row_index("row_idx")
    )

    # ------------------------------------------------------------
    # 3) Load svara annotations
    # ------------------------------------------------------------
    if annotation_path is None:
        # Adjust filename pattern here if needed for your corpus
        annotation_path = (
            corpus_root
            / recording_id
            / "raw"
            / f"{recording_id}_ann_svara.tsv"
        )

    df_svaras = load_annotations(
        file_path=annotation_path,
        annotation_type="svara",
        engine="polars",
    )

    # ------------------------------------------------------------
    # 4) Build embeddings per annotation row
    # ------------------------------------------------------------
    df_embeddings = compute_structural_embeddings_by_annotation(
        df_pitch=df_pitch,
        df_peaks=df_peaks,
        df_svaras=df_svaras,
        recording_id=recording_id,
        max_segments=max_segments,
        tau_init_sil=tau_init_sil,
    )

    # ------------------------------------------------------------
    # 5) Save
    # ------------------------------------------------------------
    if out_path is None:
        out_path = Path(
            f"data/interim/{recording_id}/features/"
            f"{recording_id}_svara_structural_embeddings.parquet"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_embeddings.write_parquet(out_path)

    print(f"Structural embeddings saved: {out_path}")
    return df_embeddings


def structural_embedding_all_recordings(
    tonic_map: dict[str, float],
    recording_ids: list[str] = SARASUDA_VARNAM,
    corpus_root: Path | str = "data/corpus",
    interim_root: Path | str = "data/interim",
    max_segments: int = 12,
    tau_init_sil: float = 0.30,
):
    """
    Run structural embedding extraction for multiple recordings.
    """
    for recording_id in recording_ids:
        print(f"Structural embedding extraction: {recording_id}...")
        structural_embedding_one_recording(
            recording_id=recording_id,
            tonic_hz=tonic_map[recording_id],
            corpus_root=corpus_root,
            interim_root=interim_root,
            max_segments=max_segments,
            tau_init_sil=tau_init_sil,
        )

    print("\nAll recordings structurally embedded.")



def compute_structural_embeddings_by_annotation(
    df_pitch: pl.DataFrame,
    df_peaks: pl.DataFrame,
    df_svaras: pl.DataFrame,
    recording_id: str,
    max_segments: int,
    tau_init_sil: float,
) -> pl.DataFrame:
    """
    Compute one structural embedding per annotated svara.

    This mirrors the logic of compute_svara_segment_features(...):
    one output row per annotation interval.
    """

    required_cols = [
        "time_rel_sec",
        "f0_savgol_p3_w13",
        "f0_savgol_p3_w13_cents",
        "flat_region",
        "row_idx",
    ]
    for c in required_cols:
        if c not in df_pitch.columns:
            raise ValueError(f"df_pitch is missing required column '{c}'")

    required_ann_cols = ["start_time_sec", "end_time_sec", "svara_label"]
    for c in required_ann_cols:
        if c not in df_svaras.columns:
            raise ValueError(f"df_svaras is missing required column '{c}'")

    rows = []

    # Precompute global peak map once
    peak_row_map = map_peaks_to_global_rows(df_pitch, df_peaks)
    t_all = df_pitch["time_rel_sec"].to_numpy()

    for segment_id, ann in enumerate(df_svaras.iter_rows(named=True)):
        t_start = float(ann["start_time_sec"])
        t_end = float(ann["end_time_sec"])

        if t_end < t_start:
            t_start, t_end = t_end, t_start

        svara_label = ann["svara_label"]

        # Slice pitch by annotation interval
        mask = (t_all >= t_start) & (t_all <= t_end)
        df_svara = df_pitch.filter(pl.Series(mask))

        if df_svara.is_empty():
            continue

        # Assign per-sample coarse labels with precedence SIL > CP > STA
        df_svara = label_samples_sil_cp_sta(df_svara)

        # Restrict global peaks to this svara and convert to local row positions
        local_peak_map = restrict_peaks_to_slice(df_svara, peak_row_map)

        # Build structural segments inside this svara
        segments = build_segments_for_one_svara(
            df_svara=df_svara,
            local_peak_map=local_peak_map,
        )

        # Assign one representative cents value per segment
        segments = assign_segment_cents(
            segments=segments,
            df_svara=df_svara,
            df_full=df_pitch,
            tau_init_sil=tau_init_sil,
            local_peak_map=local_peak_map,
        )

        # Encode to fixed-length vector
        embedding = encode_segments(
            segments=segments,
            df_svara=df_svara,
            max_segments=max_segments,
        )

        row = {
            "recording_id": recording_id,
            "segment_id": segment_id,
            "svara_label": svara_label,
            "t_start": t_start,
            "t_end": t_end,
            "duration_sec": float(t_end - t_start),
            "n_rows": int(df_svara.height),
            "n_segments": int(len(segments)),
            "embedding": embedding,
        }
        rows.append(row)

    return pl.DataFrame(rows)


def label_samples_sil_cp_sta(df_svara: pl.DataFrame) -> pl.DataFrame:
    """
    Assign per-sample labels with precedence:
        SIL > CP > STA

    SIL: f0_savgol_p3_w13 is NaN
    CP: flat_region == True
    STA: remaining voiced non-flat samples
    """

    is_sil = pl.col("f0_savgol_p3_w13").is_null() | pl.col("f0_savgol_p3_w13").is_nan()
    is_cp = (~is_sil) & pl.col("flat_region")

    return df_svara.with_columns(
        pl.when(is_sil).then(pl.lit("SIL"))
        .when(is_cp).then(pl.lit("CP"))
        .otherwise(pl.lit("STA"))
        .alias("sample_label")
    )


def map_peaks_to_global_rows(
    df_pitch: pl.DataFrame,
    df_peaks: pl.DataFrame,
) -> dict[int, float]:
    """
    Map each refined peak to the nearest global row index in df_pitch.

    We use time_savgol together with value_savgol_cents, because they refer
    to the same refined extremum location.
    """
    if df_peaks.is_empty():
        return {}

    t = df_pitch["time_rel_sec"].to_numpy()
    peak_times = df_peaks["time_savgol"].to_numpy()
    peak_vals = df_peaks["value_savgol_cents"].to_numpy()

    out = {}
    for tp, val in zip(peak_times, peak_vals):
        i = int(np.argmin(np.abs(t - tp)))
        out[int(df_pitch["row_idx"][i])] = float(val)

    return out


def restrict_peaks_to_slice(
    df_svara: pl.DataFrame,
    peak_row_map: dict[int, float],
) -> dict[int, float]:
    """
    Restrict global peaks to one svara slice and convert them to local row positions.

    Returns:
        {local_row_idx: value_savgol_cents}
    """
    global_rows = df_svara["row_idx"].to_list()
    global_to_local = {g: i for i, g in enumerate(global_rows)}

    out: dict[int, float] = {}
    for g, val in peak_row_map.items():
        if g in global_to_local:
            out[global_to_local[g]] = val

    return out



def build_segments_for_one_svara(
    df_svara: pl.DataFrame,
    local_peak_map: dict[int, float],
) -> list[dict]:
    """
    Build structural segments for one svara.

    Important rule:
    Peaks start new STA segments.
    Therefore, if a peak occurs at local index i, index i belongs to the new STA.
    """

    labels = df_svara["sample_label"].to_numpy()
    N = len(labels)

    segments: list[dict] = []
    i = 0

    while i < N:
        label = labels[i]

        # --------------------------------------------------
        # SIL run
        # --------------------------------------------------
        if label == "SIL":
            start = i
            while i < N and labels[i] == "SIL":
                i += 1

            segments.append({
                "type": "SIL",
                "start": start,
                "end": i,   # Python-style exclusive end
            })
            continue

        # --------------------------------------------------
        # CP run
        # --------------------------------------------------
        if label == "CP":
            start = i
            while i < N and labels[i] == "CP":
                i += 1

            segments.append({
                "type": "CP",
                "start": start,
                "end": i,
            })
            continue

        # --------------------------------------------------
        # STA segment
        # --------------------------------------------------
        # We are at a voiced non-flat sample.
        # This must start an STA.
        start = i
        i += 1

        while i < N:
            # Stop if we encounter silence
            if labels[i] == "SIL":
                break

            # Stop if we encounter flat region
            if labels[i] == "CP":
                break

            # Stop if current index is a peak,
            # because that peak belongs to the NEXT STA
            if i in local_peak_map:
                break

            i += 1

        segments.append({
            "type": "STA",
            "start": start,
            "end": i,
        })

    return segments



def assign_segment_cents(
    segments: list[dict],
    df_svara: pl.DataFrame,
    df_full: pl.DataFrame,
    tau_init_sil: float,
    local_peak_map: dict[int, float],
) -> list[dict]:
    """
    Assign one representative cents value to each segment.

    Segment pitch conventions:
    - CP: median pitch of the flat region
    - SIL: inherited boundary pitch
    - STA: boundary-based pitch:
        * after SIL -> first pitch of STA
        * after CP  -> last pitch of CP
        * at peak   -> peak value_savgol_cents
    """

    cents = df_svara["f0_savgol_p3_w13_cents"].to_numpy()
    times = df_svara["time_rel_sec"].to_numpy()
    global_rows = df_svara["row_idx"].to_numpy()
    full_cents = df_full["f0_savgol_p3_w13_cents"].to_numpy()

    for k, seg in enumerate(segments):
        typ = seg["type"]
        s = seg["start"]
        e = seg["end"]

        # --------------------------------------------------
        # CP
        # --------------------------------------------------
        if typ == "CP":
            vals = cents[s:e]
            vals = vals[np.isfinite(vals)]
            seg["cents"] = float(np.median(vals)) if len(vals) else np.nan
            continue

        # --------------------------------------------------
        # SIL
        # --------------------------------------------------
        if typ == "SIL":

            # Non-initial SIL inherits the last voiced pitch before silence
            if k > 0:
                prev_seg = segments[k - 1]
                prev_idx = prev_seg["end"] - 1
                seg["cents"] = float(cents[prev_idx])
                continue

            # Initial SIL: duration-dependent rule
            if e > s:
                dur_sec = float(times[e - 1] - times[s])
            else:
                dur_sec = 0.0

            # Short initial silence -> inherit previous global voiced pitch
            if dur_sec < tau_init_sil:
                g = int(global_rows[s]) - 1
                found = False

                while g >= 0:
                    v = full_cents[g]
                    if np.isfinite(v):
                        seg["cents"] = float(v)
                        found = True
                        break
                    g -= 1

                if found:
                    continue

            # Long initial silence -> use first following voiced pitch
            j = e
            found = False
            while j < len(cents):
                v = cents[j]
                if np.isfinite(v):
                    seg["cents"] = float(v)
                    found = True
                    break
                j += 1

            if found:
                continue

            seg["cents"] = np.nan
            continue

        # --------------------------------------------------
        # STA
        # --------------------------------------------------
        if typ == "STA":

            # If this STA starts on a peak, use the peak raw-cents value
            if s in local_peak_map:
                seg["cents"] = float(local_peak_map[s])
                continue

            # Initial STA -> first value
            if k == 0:
                seg["cents"] = float(cents[s])
                continue

            prev_seg = segments[k - 1]
            prev_type = prev_seg["type"]

            # SIL -> STA
            if prev_type == "SIL":
                seg["cents"] = float(cents[s])
                continue

            # CP -> STA
            if prev_type == "CP":
                seg["cents"] = float(cents[prev_seg["end"] - 1])
                continue

            # Fallback
            seg["cents"] = float(cents[s])
            continue

    return segments




def encode_segments(
    segments: list[dict],
    df_svara: pl.DataFrame,
    max_segments: int = 12,
) -> list[float]:
    """
    Convert one svara's segment list into a fixed-length vector.

    Final format:
        [total_duration, s_1, s_2, ..., s_K, padding]

    Segment format:
        [onehot_1, onehot_2, onehot_3, duration_rel, cents]
    """

    times = df_svara["time_rel_sec"].to_numpy()
    total_duration = float(times[-1] - times[0]) if len(times) > 1 else 0.0

    vec = [total_duration]

    used_segments = segments[:max_segments]

    for seg in used_segments:
        s = seg["start"]
        e = seg["end"]

        if e > s:
            dur_sec = float(times[e - 1] - times[s])
        else:
            dur_sec = 0.0

        dur_rel = dur_sec / total_duration if total_duration > 0 else 0.0

        vec.extend(
            TYPE_TO_ONEHOT[seg["type"]] +
            [dur_rel, float(seg["cents"])]
        )

    dim_per_seg = 5
    n_pad = max_segments - len(used_segments)
    vec.extend([0.0] * (n_pad * dim_per_seg))

    return vec



# -------------------------------------------------------------------
# USE EXAMPLE
# -------------------------------------------------------------------
#
#df_emb = structural_embedding_one_recording(
#    recording_id="srs_v1_bdn_sav",
#    tonic_hz=146.83,
#    corpus_root="data/corpus",
#    interim_root="data/interim",
#    max_segments=12,
#    tau_init_sil=0.30,
#)