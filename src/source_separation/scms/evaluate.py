"""
Evaluate the effect of source separation on FTA-Net pitch extraction quality.

Compares FTA-Net predictions (mir_eval melody metrics) across three audio sources
on the SCMS Carnatic test set:
    original  — mixed audio (voice + tanpura)
    as        — BS-RoFormer separated voice
    unet      — U-Net separated voice (in-house, aborted; included for reference)

Run in the `bss` environment:
    python -m src.source_separation.scms.evaluate
    python -m src.source_separation.scms.evaluate --sources original as

Metrics (mir_eval):
    VR   Voicing Recall
    VFA  Voicing False Alarm
    RPA  Raw Pitch Accuracy  (within 50 ¢, correct voicing)
    RCA  Raw Chroma Accuracy (within 50 ¢, octave-invariant)
    OA   Overall Accuracy

Paper baselines (Plaja-Roglans et al. 2023):
    FTA-C:   VR=96.35  VFA=8.38   RPA=90.17  RCA=90.46  OA=90.99
    Melodia: VR=85.75  VFA=17.17  RPA=77.51  RCA=79.81  OA=77.07
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import mir_eval

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import settings
from src.pitch_extraction.swiftf0_finetune.dataset import scms_official_split

SCMS_ROOT  = settings.PROJECT_ROOT / "data" / "datasets" / "scms"
PITCH_ROOT = settings.INTERIM_PITCH_SCMS / "ftanet"

PAPER_BASELINE = {
    "FTA-C":   dict(VR=96.35, VFA=8.38,  RPA=90.17, RCA=90.46, OA=90.99),
    "Melodia": dict(VR=85.75, VFA=17.17, RPA=77.51, RCA=79.81, OA=77.07),
}

ALL_SOURCES = ["original", "as", "unet"]
SOURCE_LABEL = {
    "original": "FTA-C / original",
    "as":       "FTA-C / BS-RoFormer",
    "unet":     "FTA-C / U-Net",
}
THRESHOLDS_SWEEP = [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


def load_annotation(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    times, freqs = [], []
    with open(csv_path) as f:
        for row in csv.reader(f):
            times.append(float(row[0].strip()))
            freqs.append(float(row[1].strip()))
    return np.array(times, dtype=np.float64), np.array(freqs, dtype=np.float64)


def load_prediction(stem: str, source: str) -> tuple[np.ndarray, np.ndarray] | None:
    path = PITCH_ROOT / f"{stem}_{source}_ftanet_raw.npy"
    if not path.exists():
        return None
    arr = np.load(path)          # (N, 2): [time_sec, f0_Hz]
    return arr[:, 0], arr[:, 1]


def evaluate_source(
    stems:   list[str],
    ann_dir: Path,
    source:  str,
) -> dict[str, float] | None:
    keys = ["Voicing Recall", "Voicing False Alarm",
            "Raw Pitch Accuracy", "Raw Chroma Accuracy", "Overall Accuracy"]
    accum = {k: [] for k in keys}
    n_skipped = 0

    for stem in stems:
        ann_path = ann_dir / f"{stem}.csv"
        if not ann_path.exists():
            continue
        pred = load_prediction(stem, source)
        if pred is None:
            n_skipped += 1
            continue

        ref_t, ref_f = load_annotation(ann_path)
        est_t, est_f = pred

        try:
            scores = mir_eval.melody.evaluate(ref_t, ref_f, est_t, est_f)
            for k in keys:
                accum[k].append(scores[k])
        except Exception:
            pass

    if n_skipped == len(stems):
        return None
    if n_skipped:
        print(f"  [warn] {source}: {n_skipped}/{len(stems)} stems missing")

    return {k: float(np.mean(v)) * 100 for k, v in accum.items() if v}


def print_row(label: str, scores: dict, note: str = ""):
    vr  = scores.get("Voicing Recall",     scores.get("VR",  0))
    vfa = scores.get("Voicing False Alarm", scores.get("VFA", 0))
    rpa = scores.get("Raw Pitch Accuracy",  scores.get("RPA", 0))
    rca = scores.get("Raw Chroma Accuracy", scores.get("RCA", 0))
    oa  = scores.get("Overall Accuracy",    scores.get("OA",  0))
    print(f"  {label:<30}  "
          f"{vr:6.2f}  {vfa:6.2f}  {rpa:6.2f}  {rca:6.2f}  {oa:6.2f}  {note}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", choices=ALL_SOURCES,
                        default=ALL_SOURCES,
                        help="Sources to evaluate (default: all three)")
    parser.add_argument("--scms-root", default=None)
    args = parser.parse_args()

    scms_root = Path(args.scms_root) if args.scms_root else SCMS_ROOT
    _, test_stems = scms_official_split(scms_root)
    ann_dir = scms_root / "annotations" / "melody"

    test_items = [s for s in test_stems
                  if (ann_dir / f"{s}.csv").exists()]
    print(f"[eval] Test stems: {len(test_items)}\n")

    header = f"  {'Model':<30}  {'VR':>6}  {'VFA':>6}  {'RPA':>6}  {'RCA':>6}  {'OA':>6}"
    sep    = "─" * len(header)

    print(f"{sep}\n{header}\n{sep}")
    for name, b in PAPER_BASELINE.items():
        print_row(name, b, note="← paper")
    print(sep)

    for source in args.sources:
        label = SOURCE_LABEL[source]
        scores = evaluate_source(test_items, ann_dir, source)
        if scores is None:
            print(f"  {label:<30}  — no predictions found, run ftanet.py first")
        else:
            print_row(label, scores)
    print(sep)


if __name__ == "__main__":
    main()
