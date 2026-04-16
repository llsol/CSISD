"""
GRU+VAE dataset builder.

Converts annotated svaras into variable-length sequences of shape (n_segments, 6)
for training the GRU+VAE model.

Segment vector format  (INPUT_DIM = 6):
    [onehot_CP, onehot_SIL, onehot_STA, dur_rel, total_dur_sec, cents_norm]

    onehot_*      : type encoding — CP=[1,0,0]  SIL=[0,1,0]  STA=[0,0,1]
    dur_rel       : segment_duration / svara_total_duration  (sum ~1 per svara)
    total_dur_sec : svara duration in seconds  (same value repeated every row)
    cents_norm    : cents_rel_tonic / 1200     (0 = tonic, 1 = octave)

Pitch assignment per segment type (delegates to assign_segment_cents):
    CP  -> median of the flat region
    SIL -> inherited from the nearest voiced boundary pitch
    STA -> peak value if the segment starts on a peak; otherwise boundary pitch
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader

import settings as S
from src.io.pitch_io import load_preprocessed_pitch, load_flat_regions, load_peaks
from src.io.annotation_io import load_annotations
from src.features.structural_embedding import (
    TYPE_TO_ONEHOT,
    label_samples_sil_cp_sta,
    map_peaks_to_global_rows,
    restrict_peaks_to_slice,
    build_segments_for_one_svara,
    assign_segment_cents,
)

INPUT_DIM = 6


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------

def segments_to_sequence(
    segments: list[dict],
    df_svara: pl.DataFrame,
) -> np.ndarray:
    """
    Convert a list of segment dicts into a (n_segments, INPUT_DIM) array.

    Each row:
        [onehot_CP, onehot_SIL, onehot_STA, dur_rel, total_dur_sec, cents_norm]
    """
    times = df_svara["time_rel_sec"].to_numpy()
    N = len(times)

    # Sample interval (used to extend the last segment to its true end boundary)
    dt = float(times[1] - times[0]) if N > 1 else 0.01

    # total_dur includes the full duration of the last sample (exclusive-end convention),
    # so that sum(dur_sec over all segments) == total_dur exactly.
    total_dur = float(times[-1] - times[0] + dt) if N > 1 else 0.0

    rows = []
    for seg in segments:
        s = seg["start"]
        e = seg["end"]

        # Exclusive-end boundary: start of the sample just after this segment.
        # This ensures t0 in the plot reconstructs to exactly times[s].
        end_time = float(times[e]) if e < N else float(times[N - 1] + dt)
        dur_sec = end_time - float(times[s]) if e > s else 0.0
        dur_rel = dur_sec / total_dur if total_dur > 0.0 else 0.0

        cents = seg.get("cents", np.nan)
        cents_norm = float(cents) / 1200.0 if np.isfinite(cents) else 0.0

        vec = TYPE_TO_ONEHOT[seg["type"]] + [dur_rel, total_dur, cents_norm]
        rows.append(vec)

    return np.array(rows, dtype=np.float32)  # (n_segments, INPUT_DIM)


# ---------------------------------------------------------------------------
# Per-recording builder
# ---------------------------------------------------------------------------

def build_svara_sequences(
    recording_id: str,
    tonic_hz: float,
    corpus_root: Path | str = "data/corpus",
    interim_root: Path | str = "data/interim",
    annotation_path: Path | str | None = None,
    tau_init_sil: float = 0.30,
) -> list[dict]:
    """
    Build one sequence per annotated svara for a single recording.

    Returns a list of dicts:
        {
            "recording_id" : str,
            "segment_id"   : int,
            "svara_label"  : str,
            "t_start"      : float,
            "t_end"        : float,
            "sequence"     : np.ndarray  shape (n_segments, INPUT_DIM)
        }
    """
    corpus_root = Path(corpus_root)
    interim_root = Path(interim_root)

    # Load data
    df_pitch = load_preprocessed_pitch(
        recording_id=recording_id,
        root_dir=interim_root,
        tonic_hz=tonic_hz,
        convert_to_cents=True,
    )
    df_flat = load_flat_regions(recording_id=recording_id, root_dir=interim_root)
    df_peaks = load_peaks(recording_id=recording_id, root_dir=interim_root)

    df_pitch = (
        df_pitch
        .join(df_flat.select(["time_rel_sec", "flat_region"]), on="time_rel_sec", how="left")
        .with_columns(pl.col("flat_region").fill_null(False))
        .with_row_index("row_idx")
    )

    if annotation_path is None:
        annotation_path = (
            corpus_root / recording_id / "raw" / f"{recording_id}_ann_svara.tsv"
        )

    df_svaras = load_annotations(
        file_path=annotation_path,
        annotation_type="svara",
        engine="polars",
    )

    # Precompute global peak map once
    peak_row_map = map_peaks_to_global_rows(df_pitch, df_peaks)
    t_all = df_pitch["time_rel_sec"].to_numpy()

    results = []

    for segment_id, ann in enumerate(df_svaras.iter_rows(named=True)):
        t_start = float(ann["start_time_sec"])
        t_end = float(ann["end_time_sec"])
        if t_end < t_start:
            t_start, t_end = t_end, t_start

        mask = (t_all >= t_start) & (t_all <= t_end)
        df_svara = df_pitch.filter(pl.Series(mask))
        if df_svara.is_empty():
            continue

        df_svara = label_samples_sil_cp_sta(df_svara)
        local_peak_map = restrict_peaks_to_slice(df_svara, peak_row_map)
        segments = build_segments_for_one_svara(df_svara, local_peak_map)
        segments = assign_segment_cents(
            segments=segments,
            df_svara=df_svara,
            df_full=df_pitch,
            tau_init_sil=tau_init_sil,
            local_peak_map=local_peak_map,
        )

        results.append({
            "recording_id": recording_id,
            "segment_id":   segment_id,
            "svara_label":  ann["svara_label"],
            "t_start":      t_start,
            "t_end":        t_end,
            "sequence":     segments_to_sequence(segments, df_svara),
        })

    return results


# ---------------------------------------------------------------------------
# Corpus-level builder
# ---------------------------------------------------------------------------

def build_corpus_sequences(
    tonic_map: dict[str, float] = S.SARASUDA_TONICS,
    recording_ids: list[str] = S.SARASUDA_VARNAM,
    corpus_root: Path | str = "data/corpus",
    interim_root: Path | str = "data/interim",
    tau_init_sil: float = 0.30,
) -> list[dict]:
    """
    Build sequences for all recordings.
    Returns a flat list of svara dicts (same format as build_svara_sequences).
    """
    all_results = []
    for rid in recording_ids:
        print(f"  {rid}...")
        all_results.extend(build_svara_sequences(
            recording_id=rid,
            tonic_hz=tonic_map[rid],
            corpus_root=corpus_root,
            interim_root=interim_root,
            tau_init_sil=tau_init_sil,
        ))

    print(f"Total svaras: {len(all_results)}")
    return all_results


# ---------------------------------------------------------------------------
# Batching utility
# ---------------------------------------------------------------------------

def pad_sequences(
    sequences: list[np.ndarray],
    max_len: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pad variable-length sequences to a common length.

    Args:
        sequences : list of (n_segments_i, INPUT_DIM) arrays
        max_len   : if None, uses the longest sequence in the list

    Returns:
        padded : (batch, max_len, INPUT_DIM)  float32, zeros for padding
        mask   : (batch, max_len)             bool, True = real data
    """
    if max_len is None:
        max_len = max(s.shape[0] for s in sequences)

    batch = len(sequences)
    padded = np.zeros((batch, max_len, INPUT_DIM), dtype=np.float32)
    mask = np.zeros((batch, max_len), dtype=bool)

    for i, seq in enumerate(sequences):
        n = min(seq.shape[0], max_len)
        padded[i, :n] = seq[:n]
        mask[i, :n] = True

    return padded, mask


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class SvaraDataset(Dataset):
    """
    PyTorch Dataset that wraps build_svara_sequences / build_corpus_sequences.

    Parameters
    ----------
    recording_ids : list[str] | None
        Recordings to include.  None → all recordings in S.SARASUDA_VARNAM.
        Pass a single-element list to train on one piece only.
    tonic_map : dict[str, float] | None
        Tonic in Hz per recording.  None → S.SARASUDA_TONICS.
    feature_cols : list[int] | None
        Column indices to keep from the 6-feature sequences before returning.
        None → return all 6 features.
        Tip: pass DATASET_FEATURE_COLS from model_gruvae to get the 5 features
        the model expects (drops total_dur_sec at index 4).
    """

    def __init__(
        self,
        recording_ids: list[str] | None = None,
        tonic_map: dict[str, float] | None = None,
        corpus_root: Path | str = "data/corpus",
        interim_root: Path | str = "data/interim",
        feature_cols: list[int] | None = None,
        tau_init_sil: float = 0.30,
    ):
        if recording_ids is None:
            recording_ids = S.SARASUDA_VARNAM
        if tonic_map is None:
            tonic_map = S.SARASUDA_TONICS

        self.feature_cols = feature_cols
        self._sequences: list[np.ndarray] = []

        for rid in recording_ids:
            svaras = build_svara_sequences(
                recording_id=rid,
                tonic_hz=tonic_map[rid],
                corpus_root=corpus_root,
                interim_root=interim_root,
                tau_init_sil=tau_init_sil,
            )
            for s in svaras:
                self._sequences.append(s["sequence"])   # (n_segments, 6)

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        seq = self._sequences[idx]                      # (n_segments, 6)
        if self.feature_cols is not None:
            seq = seq[:, self.feature_cols]             # (n_segments, n_feats)
        return torch.from_numpy(seq), seq.shape[0]      # tensor, length


def collate_svara_batch(
    batch: list[tuple[torch.Tensor, int]],
    max_len: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function for DataLoader.

    Returns
    -------
    padded  : (batch, max_len, n_feats)  float32
    lengths : (batch,)                   int64
    """
    seqs, lengths = zip(*batch)
    lengths_t = torch.tensor(lengths, dtype=torch.long)

    if max_len is None:
        max_len = int(lengths_t.max().item())

    n_feats = seqs[0].shape[1]
    padded = torch.zeros(len(seqs), max_len, n_feats, dtype=torch.float32)
    for i, (seq, length) in enumerate(zip(seqs, lengths)):
        padded[i, :length] = seq[:length]

    return padded, lengths_t


def build_dataloader(
    recording_ids: list[str] | None = None,
    tonic_map: dict[str, float] | None = None,
    feature_cols: list[int] | None = None,
    batch_size: int = 32,
    shuffle: bool = True,
    max_len: int | None = None,
    corpus_root: Path | str = "data/corpus",
    interim_root: Path | str = "data/interim",
) -> DataLoader:
    """
    Build a DataLoader for one or all recordings.

    Examples
    --------
    # All corpus
    loader = build_dataloader()

    # Single piece
    loader = build_dataloader(recording_ids=["srs_v1_bdn_sav"])

    # With model feature selection (5 features)
    from src.models.gruvae.model_gruvae import DATASET_FEATURE_COLS
    loader = build_dataloader(feature_cols=DATASET_FEATURE_COLS)
    """
    dataset = SvaraDataset(
        recording_ids=recording_ids,
        tonic_map=tonic_map,
        corpus_root=corpus_root,
        interim_root=interim_root,
        feature_cols=feature_cols,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda b: collate_svara_batch(b, max_len=max_len),
    )
