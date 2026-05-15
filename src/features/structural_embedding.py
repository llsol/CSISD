from __future__ import annotations

from pathlib import Path
import numpy as np
import polars as pl

from settings import RECORDING_SELECTION
from src.io.pitch_io import load_preprocessed_pitch, load_flat_regions, load_peaks
from src.io.annotation_io import load_annotations


# -------------------------------------------------------------------
# Fixed encoding for segment types
# -------------------------------------------------------------------
TYPE_TO_ONEHOT = {
    "CP":   [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "SIL":  [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    "STAp": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # STA ending at local maximum
    "STAt": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # STA ending at local minimum
    "TRa":  [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # TR ascending (cents_end > cents_start)
    "TRd":  [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # TR descending (cents_end < cents_start)
}


# -------------------------------------------------------------------
# PUBLIC API
# -------------------------------------------------------------------
def structural_embedding_one_recording(
    recording_id: str,
    tonic_hz: float,
    corpus_root: Path | str = "data/corpus",
    interim_root: Path | str | None = None,
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

    import settings as _S
    corpus_root  = Path(corpus_root)
    interim_root = _S.INTERIM_RECORDINGS if interim_root is None else Path(interim_root)

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
            f"{_S.INTERIM_RECORDINGS}/{recording_id}/features/{recording_id}_svara_structural_embeddings.parquet"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_embeddings.write_parquet(out_path)

    print(f"Structural embeddings saved: {out_path}")
    return df_embeddings


def structural_embedding_all_recordings(
    tonic_map: dict[str, float],
    recording_ids: list[str] = RECORDING_SELECTION,
    corpus_root: Path | str = "data/corpus",
    interim_root: Path | str | None = None,
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
) -> dict[int, tuple[float, str]]:
    """
    Map each refined peak to the nearest global row index in df_pitch.

    Returns:
        {global_row_idx: (value_savgol_cents, extremum_kind)}
        where extremum_kind is 'max' or 'min'.
    """
    if df_peaks.is_empty():
        return {}

    t = df_pitch["time_rel_sec"].to_numpy()
    peak_times = df_peaks["time_savgol"].to_numpy()
    peak_vals  = df_peaks["value_savgol_cents"].to_numpy()
    peak_kinds = df_peaks["extremum_kind"].to_numpy()

    out = {}
    for tp, val, kind in zip(peak_times, peak_vals, peak_kinds):
        i = int(np.argmin(np.abs(t - tp)))
        out[int(df_pitch["row_idx"][i])] = (float(val), str(kind))

    return out


def restrict_peaks_to_slice(
    df_svara: pl.DataFrame,
    peak_row_map: dict[int, tuple[float, str]],
) -> dict[int, tuple[float, str]]:
    """
    Restrict global peaks to one svara slice and convert them to local row positions.

    Returns:
        {local_row_idx: (value_savgol_cents, extremum_kind)}
    """
    global_rows = df_svara["row_idx"].to_list()
    global_to_local = {g: i for i, g in enumerate(global_rows)}

    out: dict[int, tuple[float, str]] = {}
    for g, val in peak_row_map.items():
        if g in global_to_local:
            out[global_to_local[g]] = val

    return out



def build_segments_for_one_svara(
    df_svara: pl.DataFrame,
    local_peak_map: dict[int, tuple[float, str]],
) -> list[dict]:
    """
    Build structural segments for one svara.

    Segment types:
      CP   — voiced flat region; cents = median.
      STAp — voiced non-flat ending AT a local maximum; cents = peak value.
      STAt — voiced non-flat ending AT a local minimum; cents = trough value.
      TRa  — voiced non-flat ending at CP/SIL with ascending direction; cents = 0.
      TRd  — voiced non-flat ending at CP/SIL with descending direction; cents = 0.
      SIL  — unvoiced; cents = 0.

    The peak is the ENDPOINT of STAp/STAt.  After a STA the
    descent/ascent back to the next CP naturally becomes the next TR segment.
    """
    labels = df_svara["sample_label"].to_numpy()
    cents  = df_svara["f0_savgol_p3_w13_cents"].to_numpy()
    N = len(labels)
    segments: list[dict] = []
    i = 0

    while i < N:
        label = labels[i]

        if label == "SIL":
            start = i
            while i < N and labels[i] == "SIL":
                i += 1
            segments.append({"type": "SIL", "start": start, "end": i})
            continue

        if label == "CP":
            start = i
            while i < N and labels[i] == "CP":
                i += 1
            segments.append({"type": "CP", "start": start, "end": i})
            continue

        # Voiced non-flat: advance until CP / SIL / peak (inclusive).
        start = i
        i += 1
        peak_pos = -1

        while i < N:
            if labels[i] == "SIL":
                break
            if labels[i] == "CP":
                break
            if i in local_peak_map:
                peak_pos = i
                i += 1          # include the peak sample in this segment
                break
            i += 1

        if peak_pos >= 0:
            # Degrade to TR if the peak is immediately followed by CP or SIL:
            # no descending/ascending arc exists after the peak.
            if i < N and labels[i] in ("CP", "SIL"):
                peak_pos = -1
            else:
                _, peak_kind = local_peak_map[peak_pos]
                seg_type = "STAp" if peak_kind == "max" else "STAt"

        if peak_pos < 0:
            # TR: determine direction from start/end cents values
            seg_cents = cents[start:i]
            finite = seg_cents[np.isfinite(seg_cents)]
            if len(finite) >= 2 and finite[-1] != finite[0]:
                seg_type = "TRa" if finite[-1] > finite[0] else "TRd"
            else:
                # Too few samples to determine direction: infer from previous segment.
                # After STAp (local max) expect descent → TRd; after STAt → TRa.
                prev_type = segments[-1]["type"] if segments else None
                if prev_type == "STAp":
                    seg_type = "TRd"
                elif prev_type == "STAt":
                    seg_type = "TRa"
                else:
                    seg_type = "TRa"

        segments.append({"type": seg_type, "start": start, "end": i})

    segments = _absorb_short_tr_into_next(segments)
    return _degrade_sta_before_boundary(segments)


def _degrade_sta_before_boundary(segments: list[dict]) -> list[dict]:
    """
    Convert any STAp/STAt that is immediately followed by CP or SIL into a TR.

    Direction inferred from peak kind:
      STAp (ended at local max, was ascending) → TRa
      STAt (ended at local min, was descending) → TRd
    """
    _sta_to_tr = {"STAp": "TRa", "STAt": "TRd"}
    for i, seg in enumerate(segments):
        if seg["type"] not in _sta_to_tr:
            continue
        nxt = segments[i + 1] if i + 1 < len(segments) else None
        if nxt is not None and nxt["type"] in ("CP", "SIL"):
            seg["type"] = _sta_to_tr[seg["type"]]
    return segments


def _absorb_short_tr_into_next(
    segments: list[dict],
    min_samples: int = 3,
) -> list[dict]:
    """
    Absorb a short TR segment (< min_samples) into the following CP or SIL.
    Removes boundary artifacts between STA peaks and stable regions.
    """
    out: list[dict] = []
    i = 0
    while i < len(segments):
        seg = segments[i]
        is_short_tr = (
            seg["type"] in ("TRa", "TRd")
            and (seg["end"] - seg["start"]) < min_samples
        )
        if is_short_tr and i + 1 < len(segments):
            nxt = segments[i + 1]
            if nxt["type"] in ("CP", "SIL"):
                nxt["start"] = seg["start"]   # extend next segment backwards
                i += 1
                continue                       # skip this TR
        out.append(seg)
        i += 1
    return out



def assign_segment_cents(
    segments: list[dict],
    df_svara: pl.DataFrame,
    local_peak_map: dict[int, tuple[float, str]],
) -> list[dict]:
    """
    Assign one representative cents value to each segment.

    CP  → median of the flat region
    STA → peak value at the end of the segment
    TR  → 0.0  (transition, no absolute pitch)
    SIL → 0.0  (silence, no pitch)
    """
    cents = df_svara["f0_savgol_p3_w13_cents"].to_numpy()

    for seg in segments:
        typ = seg["type"]
        s   = seg["start"]
        e   = seg["end"]

        if typ == "CP":
            vals = cents[s:e]
            vals = vals[np.isfinite(vals)]
            seg["cents"] = float(np.median(vals)) if len(vals) else 0.0

        elif typ in ("STAp", "STAt"):
            # Peak is the last sample (e-1); find it in local_peak_map.
            peak_in_seg = {k: v for k, v in local_peak_map.items() if s <= k < e}
            if peak_in_seg:
                seg["cents"] = float(peak_in_seg[max(peak_in_seg)][0])
            else:
                # Fallback: endpoint value (should not normally happen)
                vals = cents[s:e]
                vals = vals[np.isfinite(vals)]
                seg["cents"] = float(vals[-1]) if len(vals) else 0.0

        else:  # TRa, TRd, or SIL
            seg["cents"] = 0.0

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
        [onehot_CP, onehot_SIL, onehot_STAp, onehot_STAt, onehot_TRa, onehot_TRd, dur_rel, dur_abs_sec, cents]
    """

    times = df_svara["time_rel_sec"].to_numpy()
    dt = float(times[1] - times[0]) if len(times) > 1 else 0.01
    total_duration = float(times[-1] - times[0] + dt) if len(times) > 1 else 0.0

    vec = [total_duration]

    used_segments = segments[:max_segments]

    for seg in used_segments:
        s = seg["start"]
        e = seg["end"]

        if e > s:
            end_time = float(times[e]) if e < len(times) else float(times[-1] + dt)
            dur_sec = end_time - float(times[s])
        else:
            dur_sec = 0.0

        dur_rel = dur_sec / total_duration if total_duration > 0 else 0.0

        vec.extend(
            TYPE_TO_ONEHOT[seg["type"]] +
            [dur_rel, dur_sec, float(seg["cents"])]
        )

    dim_per_seg = 9   # 6 onehot + dur_rel + dur_abs_sec + cents
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