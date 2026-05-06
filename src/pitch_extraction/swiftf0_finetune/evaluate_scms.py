"""
Evaluate a fine-tuned SwiftF0 model on the SCMS Carnatic test set.

Computes the standard melody extraction metrics (mir_eval):
    VR   — Voicing Recall
    VFA  — Voicing False Alarm
    RPA  — Raw Pitch Accuracy  (within 50 ¢ of ground truth, correct voicing)
    RCA  — Raw Chroma Accuracy (within 50 ¢, octave-invariant)
    OA   — Overall Accuracy

Optionally sweeps the confidence threshold to find the optimal value
(threshold is a free hyperparameter; default 0.9 matches swift_f0 original).

Paper baseline (Table 2, Plaja-Roglans et al. 2023, SCMS test set):
    FTA-C:   VR=96.35  VFA=8.38   RPA=90.17  RCA=90.46  OA=90.99
    Melodia: VR=85.75  VFA=17.17  RPA=77.51  RCA=79.81  OA=77.07

Usage:
    python -m src.pitch_extraction.swiftf0_finetune.evaluate_scms
    python -m src.pitch_extraction.swiftf0_finetune.evaluate_scms --model scratch
    python -m src.pitch_extraction.swiftf0_finetune.evaluate_scms --sweep
    python -m src.pitch_extraction.swiftf0_finetune.evaluate_scms --thr 0.7
    python -m src.pitch_extraction.swiftf0_finetune.evaluate_scms --run run_002
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import mir_eval

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import settings
from src.pitch_extraction.swiftf0_finetune.dataset import scms_official_split

SCMS_ROOT = settings.PROJECT_ROOT / "data" / "datasets" / "scms"

PAPER_BASELINE = {
    "FTA-C":   dict(VR=96.35, VFA=8.38,  RPA=90.17, RCA=90.46, OA=90.99),
    "Melodia": dict(VR=85.75, VFA=17.17, RPA=77.51, RCA=79.81, OA=77.07),
}

THRESHOLDS_SWEEP = [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(model_type: str, run_dir: Path | None, device: torch.device):
    if model_type == "finetune":
        from src.pitch_extraction.swiftf0_finetune.model import SwiftF0
        ckpt_dir = settings.DATA_INTERIM / "models" / "swiftf0_carnatic"
        Model = SwiftF0
    else:
        from src.pitch_extraction.swiftf0_scratch.model import SwiftF0Scratch
        ckpt_dir = settings.DATA_INTERIM / "models" / "swiftf0_scratch"
        Model = SwiftF0Scratch

    if run_dir is None:
        runs = sorted(ckpt_dir.glob("run_*"))
        if not runs:
            raise FileNotFoundError(f"No training runs in {ckpt_dir}")
        run_dir = runs[-1]

    best = run_dir / "best.pt"
    if not best.exists():
        best = run_dir / "last.pt"

    model = Model().to(device)
    ckpt  = torch.load(best, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[eval] Loaded {model_type} epoch={ckpt['epoch']}  "
          f"val_loss={ckpt.get('best_val', '?'):.4f}  ({best})")
    return model


# ── annotation loading ────────────────────────────────────────────────────────

def load_annotation(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    times, freqs = [], []
    with open(csv_path) as f:
        for row in csv.reader(f):
            times.append(float(row[0].strip()))
            freqs.append(float(row[1].strip()))
    return np.array(times, dtype=np.float64), np.array(freqs, dtype=np.float64)


# ── inference ─────────────────────────────────────────────────────────────────

def run_inference(
    model,
    wav_path: Path,
    device:   torch.device,
    sr:       int,
    hop:      int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (times_sec, pitch_hz, confidence) at model frame rate."""
    import librosa
    audio, _ = librosa.load(str(wav_path), sr=sr, mono=True)

    chunk_samples = 30 * sr
    all_times, all_pitch, all_conf = [], [], []
    time_offset = 0.0
    hop_sec     = hop / sr

    with torch.no_grad():
        for start in range(0, len(audio), chunk_samples):
            chunk  = audio[start : start + chunk_samples]
            tensor = torch.from_numpy(chunk).unsqueeze(0).to(device)
            pitch_hz, confidence, _ = model(tensor)
            pitch_hz   = pitch_hz[0].cpu().numpy()
            confidence = confidence[0].cpu().numpy()
            n = len(pitch_hz)
            all_times.append(time_offset + np.arange(n) * hop_sec)
            all_pitch.append(pitch_hz)
            all_conf.append(confidence)
            time_offset += n * hop_sec

    return (np.concatenate(all_times),
            np.concatenate(all_pitch),
            np.concatenate(all_conf))


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate_at_threshold(
    ref_times:   list[np.ndarray],
    ref_freqs:   list[np.ndarray],
    est_times:   list[np.ndarray],
    est_pitches: list[np.ndarray],
    est_confs:   list[np.ndarray],
    threshold:   float,
) -> dict[str, float]:
    """Mean mir_eval melody metrics across all test items (unweighted per item)."""
    keys = ["Voicing Recall", "Voicing False Alarm",
            "Raw Pitch Accuracy", "Raw Chroma Accuracy", "Overall Accuracy"]
    accum = {k: [] for k in keys}

    for ref_t, ref_f, est_t, est_p, est_c in zip(
            ref_times, ref_freqs, est_times, est_pitches, est_confs):
        voiced      = est_c >= threshold
        est_f_thr   = est_p.copy()
        est_f_thr[~voiced] = 0.0
        try:
            scores = mir_eval.melody.evaluate(ref_t, ref_f, est_t, est_f_thr)
            for k in keys:
                accum[k].append(scores[k])
        except Exception:
            pass

    return {k: float(np.mean(v)) * 100 for k, v in accum.items() if v}


def print_row(label: str, scores: dict[str, float], thr: str = "—", note: str = ""):
    vr  = scores.get("Voicing Recall",      scores.get("VR",  0))
    vfa = scores.get("Voicing False Alarm",  scores.get("VFA", 0))
    rpa = scores.get("Raw Pitch Accuracy",   scores.get("RPA", 0))
    rca = scores.get("Raw Chroma Accuracy",  scores.get("RCA", 0))
    oa  = scores.get("Overall Accuracy",     scores.get("OA",  0))
    print(f"  {label:<24} {thr:>5}  "
          f"{vr:6.2f}  {vfa:6.2f}  {rpa:6.2f}  {rca:6.2f}  {oa:6.2f}  {note}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["finetune", "scratch"], default="finetune")
    parser.add_argument("--run",   default=None,
                        help="Run directory name, e.g. run_002 (default: latest)")
    parser.add_argument("--thr",   type=float, default=0.9,
                        help="Confidence threshold for voicing (default: 0.9)")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep multiple thresholds to find the optimal")
    parser.add_argument("--scms-root", default=None)
    args = parser.parse_args()

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scms_root = Path(args.scms_root) if args.scms_root else SCMS_ROOT

    if args.model == "finetune":
        from src.pitch_extraction.swiftf0_finetune.model import SR, HOP
        ckpt_dir = settings.DATA_INTERIM / "models" / "swiftf0_carnatic"
    else:
        from src.pitch_extraction.swiftf0_scratch.model import SR, HOP
        ckpt_dir = settings.DATA_INTERIM / "models" / "swiftf0_scratch"

    run_dir = (ckpt_dir / args.run) if args.run else None
    model   = load_model(args.model, run_dir, device)

    _, test_stems = scms_official_split(scms_root)
    audio_dir = scms_root / "audio"
    ann_dir   = scms_root / "annotations" / "melody"

    test_items = [(audio_dir / f"{s}.wav", ann_dir / f"{s}.csv")
                  for s in test_stems
                  if (audio_dir / f"{s}.wav").exists() and (ann_dir / f"{s}.csv").exists()]
    print(f"[eval] Test items: {len(test_items)}  device={device}")

    # ── inference ─────────────────────────────────────────────────────────────
    ref_times, ref_freqs = [], []
    est_times, est_pitches, est_confs = [], [], []

    for i, (wav, csv_p) in enumerate(test_items):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"[eval] Inference {i+1}/{len(test_items)} ...", end="\r", flush=True)
        ref_t, ref_f = load_annotation(csv_p)
        est_t, est_p, est_c = run_inference(model, wav, device, SR, HOP)
        ref_times.append(ref_t);  ref_freqs.append(ref_f)
        est_times.append(est_t);  est_pitches.append(est_p);  est_confs.append(est_c)

    print(f"[eval] Inference done — {len(test_items)} chunks.              ")

    # ── results ───────────────────────────────────────────────────────────────
    model_label = f"SwiftF0-{args.model}"
    header = f"  {'Model':<24} {'thr':>5}  {'VR':>6}  {'VFA':>6}  {'RPA':>6}  {'RCA':>6}  {'OA':>6}"
    sep    = "─" * len(header)

    print(f"\n{sep}\n{header}\n{sep}")
    for name, b in PAPER_BASELINE.items():
        print_row(name, b, note="← paper")
    print(sep)

    if args.sweep:
        best_oa, best_thr, best_scores = -1.0, args.thr, {}
        for thr in THRESHOLDS_SWEEP:
            scores = evaluate_at_threshold(
                ref_times, ref_freqs, est_times, est_pitches, est_confs, thr)
            print_row(model_label, scores, thr=f"{thr:.2f}")
            if scores.get("Overall Accuracy", 0) > best_oa:
                best_oa, best_thr, best_scores = scores["Overall Accuracy"], thr, scores
        print(sep)
        print_row(f"{model_label} [best]", best_scores, thr=f"{best_thr:.2f}", note="← best OA")
    else:
        scores = evaluate_at_threshold(
            ref_times, ref_freqs, est_times, est_pitches, est_confs, args.thr)
        print_row(model_label, scores, thr=f"{args.thr:.2f}")

    print(f"{sep}\n")


if __name__ == "__main__":
    main()
