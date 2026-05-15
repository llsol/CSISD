"""
Compare all pitch extractors on the SCMS Carnatic test set.

Systems evaluated:
    FTANet          original audio  (cached predictions)
    FTANet          AS-separated
    SwiftF0-finetune original audio  (live inference)
    SwiftF0-finetune AS-separated
    SwiftF0-scratch  original audio  (live inference)
    SwiftF0-scratch  AS-separated

Metrics (mir_eval melody):
    VR   Voicing Recall
    VFA  Voicing False Alarm
    RPA  Raw Pitch Accuracy  (±50 ¢, correct voicing)
    RCA  Raw Chroma Accuracy (±50 ¢, octave-invariant)
    OA   Overall Accuracy

FTANet predictions must be pre-computed with:
    /home/lluis/anaconda3/envs/compiam/bin/python -m src.source_separation.scms.ftanet

Usage:
    python -m src.pitch_extraction.swiftf0_scratch.evaluate_scms
    python -m src.pitch_extraction.swiftf0_scratch.evaluate_scms --thr 0.8
    python -m src.pitch_extraction.swiftf0_scratch.evaluate_scms --sweep
    python -m src.pitch_extraction.swiftf0_scratch.evaluate_scms --run-scratch run_003
    python -m src.pitch_extraction.swiftf0_scratch.evaluate_scms --run-fine run_002
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import mir_eval

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import settings
from src.pitch_extraction.swiftf0_finetune.dataset import scms_official_split

RESULTS_DIR        = settings.RESULTS_DIR / "scms_eval"
SCMS_ROOT          = settings.PROJECT_ROOT / "data" / "datasets" / "scms"
SEP_ROOT           = settings.INTERIM_SEP_SCMS
PITCH_ROOT_FTANET  = settings.INTERIM_PITCH_SCMS / "ftanet"
PITCH_ROOT_FINE    = settings.INTERIM_PITCH_SCMS / "swiftf0_finetune"
PITCH_ROOT_SCRATCH = settings.INTERIM_PITCH_SCMS / "swiftf0_scratch"

PAPER_BASELINE = {
    "FTA-C (paper)": dict(VR=96.35, VFA=8.38,  RPA=90.17, RCA=90.46, OA=90.99),
    "Melodia (paper)": dict(VR=85.75, VFA=17.17, RPA=77.51, RCA=79.81, OA=77.07),
}
THRESHOLDS_SWEEP = [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]


# ── annotation ────────────────────────────────────────────────────────────────

def load_annotation(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    times, freqs = [], []
    with open(csv_path) as f:
        for row in csv.reader(f):
            times.append(float(row[0].strip()))
            freqs.append(float(row[1].strip()))
    return np.array(times, dtype=np.float64), np.array(freqs, dtype=np.float64)


# ── FTANet: load cached predictions ──────────────────────────────────────────

def load_ftanet(stem: str, source: str) -> tuple[np.ndarray, np.ndarray] | None:
    path = PITCH_ROOT_FTANET / source / f"{stem}_{source}_ftanet_raw.npy"
    if not path.exists():
        return None
    data = np.load(path)
    return data[:, 0].astype(np.float64), data[:, 1].astype(np.float64)


# ── SwiftF0: live inference ───────────────────────────────────────────────────

def load_swiftf0(model_type: str, run_name: str | None, device: torch.device):
    if model_type == "finetune":
        from src.pitch_extraction.swiftf0_finetune.model import SwiftF0
        ckpt_dir = settings.SWIFTF0_CARNATIC_DIR
        Model = SwiftF0
    else:
        from src.pitch_extraction.swiftf0_scratch.model import SwiftF0Scratch
        ckpt_dir = settings.SWIFTF0_SCRATCH_DIR
        Model = SwiftF0Scratch

    run_dir = (ckpt_dir / run_name) if run_name else sorted(ckpt_dir.glob("run_*"))[-1]
    best    = run_dir / "best.pt"
    if not best.exists():
        best = run_dir / "last.pt"

    model = Model().to(device)
    ckpt  = torch.load(best, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"  [{model_type}] epoch={ckpt['epoch']}  val={ckpt.get('best_val', '?'):.4f}  {best}")
    return model, run_dir.name

def load_cached_swiftf0(model_type: str, stem: str, source: str) -> tuple[np.ndarray, np.ndarray] | None:
    if model_type == "finetune":
        root = PITCH_ROOT_FINE
        name = f"{stem}_{source}_swiftf0-finetune_raw.npy"
    elif model_type == "scratch":
        root = PITCH_ROOT_SCRATCH
        name = f"{stem}_{source}_swiftf0-scratch_raw.npy"
    else:
        raise ValueError(model_type)

    path = root / source / name
    if not path.exists():
        return None

    data = np.load(path)
    return data[:, 0].astype(np.float64), data[:, 1].astype(np.float64)

def infer_swiftf0(
    model,
    wav_path: Path,
    device:   torch.device,
    sr:       int,
    hop:      int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import librosa
    audio, _ = librosa.load(str(wav_path), sr=sr, mono=True)
    chunk_samples = 30 * sr
    all_t, all_p, all_c = [], [], []
    t_offset = 0.0
    hop_sec  = hop / sr
    with torch.no_grad():
        for start in range(0, len(audio), chunk_samples):
            chunk  = audio[start : start + chunk_samples]
            tensor = torch.from_numpy(chunk).unsqueeze(0).to(device)
            pitch_hz, conf, _ = model(tensor)
            pitch_hz = pitch_hz[0].cpu().numpy()
            conf     = conf[0].cpu().numpy()
            n = len(pitch_hz)
            all_t.append(t_offset + np.arange(n) * hop_sec)
            all_p.append(pitch_hz)
            all_c.append(conf)
            t_offset += n * hop_sec
    return np.concatenate(all_t), np.concatenate(all_p), np.concatenate(all_c)


def swiftf0_audio_path(stem: str, source: str) -> Path:
    if source == "original":
        return SCMS_ROOT / "audio" / f"{stem}.wav"
    elif source == "as":
        return SEP_ROOT / stem / f"{stem}_as_voice.wav"
    elif source == "unet":
        return SEP_ROOT / stem / f"{stem}_unet_voice.wav"
    raise ValueError(source)


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(
    ref_times:   list[np.ndarray],
    ref_freqs:   list[np.ndarray],
    est_times:   list[np.ndarray],
    est_freqs:   list[np.ndarray],
) -> dict[str, float]:
    keys = ["Voicing Recall", "Voicing False Alarm",
            "Raw Pitch Accuracy", "Raw Chroma Accuracy", "Overall Accuracy"]
    accum = {k: [] for k in keys}
    for ref_t, ref_f, est_t, est_f in zip(ref_times, ref_freqs, est_times, est_freqs):
        try:
            scores = mir_eval.melody.evaluate(ref_t, ref_f, est_t, est_f)
            for k in keys:
                accum[k].append(scores[k])
        except Exception:
            pass
    return {k: float(np.mean(v)) * 100 for k, v in accum.items() if v}


def apply_threshold(pitch: np.ndarray, conf: np.ndarray, thr: float) -> np.ndarray:
    out = pitch.copy()
    has_conf = ~np.isnan(conf)
    out[has_conf & (conf < thr)] = 0.0
    return out


def print_header():
    h = f"  {'System':<26} {'src':<8}  {'VR':>6}  {'VFA':>6}  {'RPA':>6}  {'RCA':>6}  {'OA':>6}"
    sep = "─" * len(h)
    print(f"\n{sep}\n{h}\n{sep}")
    return sep


def print_row(label: str, source: str, scores: dict, note: str = ""):
    vr  = scores.get("Voicing Recall",     scores.get("VR",  0))
    vfa = scores.get("Voicing False Alarm", scores.get("VFA", 0))
    rpa = scores.get("Raw Pitch Accuracy",  scores.get("RPA", 0))
    rca = scores.get("Raw Chroma Accuracy", scores.get("RCA", 0))
    oa  = scores.get("Overall Accuracy",    scores.get("OA",  0))
    print(f"  {label:<26} {source:<8}  {vr:6.2f}  {vfa:6.2f}  {rpa:6.2f}  {rca:6.2f}  {oa:6.2f}  {note}")


# ── logging ───────────────────────────────────────────────────────────────────

def save_results(
    args,
    run_scratch: str,
    run_fine: str,
    n_stems: int,
    results: list[dict],
) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    thr_tag = "sweep" if (args.sweep or args.sweep_all) else f"thr{args.thr:.2f}"
    fname = f"scms_eval_{ts}_{thr_tag}.json"
    out = {
        "timestamp":        ts,
        "n_stems_per_source": n_stems,
        "threshold":   args.thr if not (args.sweep or args.sweep_all) else None,
        "sweep":       args.sweep or args.sweep_all,
        "sources":     args.sources,
        "run_scratch": run_scratch,
        "run_fine":    run_fine,
        "paper_baselines": PAPER_BASELINE,
        "results":     results,
    }
    path = RESULTS_DIR / fname
    path.write_text(json.dumps(out, indent=2))
    print(f"\n[eval] Results saved → {path}")
    return path


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare FTANet / SwiftF0-finetune / SwiftF0-scratch on SCMS."
    )
    parser.add_argument("--thr",          type=float, default=0.9)
    parser.add_argument("--sweep",        action="store_true")
    parser.add_argument("--sweep-all",    action="store_true",
                        help="Print scores at every threshold (implies --sweep)")
    parser.add_argument("--run-scratch",  default=None, help="SwiftF0-scratch run dir")
    parser.add_argument("--run-fine",     default=None, help="SwiftF0-finetune run dir")
    parser.add_argument("--sources",      nargs="+", default=["original", "as"],
                        choices=["original", "as", "unet"])
    parser.add_argument("--scms-root",    default=None)
    parser.add_argument("--cached", action="store_true",
                        help="Use cached SwiftF0 predictions instead of live inference.")
    args = parser.parse_args()

    scms_root = Path(args.scms_root) if args.scms_root else SCMS_ROOT
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] device={device}")

    _, test_stems = scms_official_split(scms_root)
    ann_dir = scms_root / "annotations" / "melody"
    test_stems = [s for s in test_stems if (ann_dir / f"{s}.csv").exists()]
    print(f"[eval] Test stems: {len(test_stems)}")

    # ground truth (same for all systems)
    ref_times = [load_annotation(ann_dir / f"{s}.csv")[0] for s in test_stems]
    ref_freqs = [load_annotation(ann_dir / f"{s}.csv")[1] for s in test_stems]

    # load SwiftF0 models once
    from src.pitch_extraction.swiftf0_finetune.model import SR as SR_FINE, HOP as HOP_FINE
    from src.pitch_extraction.swiftf0_scratch.model import SR as SR_SC,   HOP as HOP_SC

    print("\n[eval] Loading SwiftF0-finetune ...")
    model_fine, run_fine_name = load_swiftf0("finetune", args.run_fine, device)
    print("[eval] Loading SwiftF0-scratch ...")
    model_sc, run_scratch_name = load_swiftf0("scratch",  args.run_scratch, device)

    # collect predictions per (extractor, source)
    SYSTEMS = [
        ("FTANet",          "ftanet"),
        ("SwiftF0-finetune","finetune"),
        ("SwiftF0-scratch", "scratch"),
    ]

    # pre-run inference for all systems × sources
    predictions = {}   # (extractor, source) → list of (est_times, est_freqs, est_confs?)

    for source in args.sources:
        print(f"\n[eval] Inference — source={source}")

        # FTANet (cached)
        ftanet_t, ftanet_f = [], []
        n_missing = 0
        for stem in test_stems:
            res = load_ftanet(stem, source)
            if res is None:
                n_missing += 1
                ftanet_t.append(np.array([])); ftanet_f.append(np.array([]))
            else:
                ftanet_t.append(res[0]); ftanet_f.append(res[1])
        if n_missing:
            print(f"  [FTANet/{source}] {n_missing}/{len(test_stems)} stems missing cache")
        predictions[("ftanet", source)] = (ftanet_t, ftanet_f, None)

        # SwiftF0-finetune and SwiftF0-scratch: cached → live fallback per stem
        for (model_type, model, sr, hop, label) in [
            ("finetune", model_fine, SR_FINE, HOP_FINE, "finetune"),
            ("scratch",  model_sc,  SR_SC,   HOP_SC,   "scratch"),
        ]:
            est_t, est_p, est_c = [], [], []
            n_cached = n_live = n_missing = 0
            for i, stem in enumerate(test_stems):
                res = None if (args.sweep or args.sweep_all) else load_cached_swiftf0(model_type, stem, source)
                if res is not None:
                    t, p = res
                    est_t.append(t)
                    est_p.append(p)
                    est_c.append(np.full(len(p), np.nan))
                    n_cached += 1
                elif args.cached:
                    est_t.append(np.array([])); est_p.append(np.array([])); est_c.append(np.array([]))
                    n_missing += 1
                else:
                    wav = swiftf0_audio_path(stem, source)
                    if not wav.exists():
                        est_t.append(np.array([])); est_p.append(np.array([])); est_c.append(np.array([]))
                        n_missing += 1
                        continue
                    if (i + 1) % 20 == 0 or i == 0:
                        print(f"  [{label}/{source}] {i+1}/{len(test_stems)}", end="\r", flush=True)
                    t, p, c = infer_swiftf0(model, wav, device, sr, hop)
                    est_t.append(t); est_p.append(p); est_c.append(c)
                    n_live += 1

            parts = [f"cached={n_cached}", f"live={n_live}"]
            if n_missing:
                parts.append(f"missing={n_missing}")
            print(f"  [{label}/{source}] {', '.join(parts)}        ")
            predictions[(label, source)] = (est_t, est_p, est_c)

    # discard stems missing in ≥1 system, computed per source independently
    n_total = len(test_stems)

    valid_per_source: dict[str, list[int]] = {}
    for source in args.sources:
        valid = np.ones(n_total, dtype=bool)
        for key in [s[1] for s in SYSTEMS]:
            est_t, _, _ = predictions[(key, source)]
            for i, t in enumerate(est_t):
                if len(t) == 0:
                    valid[i] = False
        keep = [i for i, v in enumerate(valid) if v]
        n_disc = n_total - len(keep)
        if n_disc:
            print(f"\n[eval] [{source}] Discarding {n_disc} stems missing in ≥1 system — "
                  f"evaluating on {len(keep)}/{n_total}")
        valid_per_source[source] = keep

    # print results
    sep = print_header()
    for name, b in PAPER_BASELINE.items():
        print_row(name, "—", b, note="← paper")
    print(sep)

    do_sweep   = args.sweep or args.sweep_all
    thresholds = THRESHOLDS_SWEEP if do_sweep else [args.thr]

    log_entries: list[dict] = []

    for source in args.sources:
        keep      = valid_per_source[source]
        src_ref_t = [ref_times[i] for i in keep]
        src_ref_f = [ref_freqs[i] for i in keep]
        src_stems = [test_stems[i] for i in keep]

        for extractor_label, key in SYSTEMS:
            raw_t, raw_p, raw_c = predictions[(key, source)]
            est_t = [raw_t[i] for i in keep]
            est_p = [raw_p[i] for i in keep]
            est_c = [raw_c[i] for i in keep] if raw_c is not None else None

            if do_sweep and est_c is not None:
                best_oa, best_scores, best_thr = -1.0, {}, args.thr
                for thr in thresholds:
                    est_f = [apply_threshold(p, c, thr) for p, c in zip(est_p, est_c)]
                    sc = evaluate(src_ref_t, src_ref_f, est_t, est_f)
                    if args.sweep_all:
                        print_row(extractor_label, source, sc, note=f"thr={thr:.2f}")
                        log_entries.append({
                            "system": extractor_label, "source": source,
                            "threshold": thr, "best": False, **sc,
                        })
                    if sc.get("Overall Accuracy", 0) > best_oa:
                        best_oa, best_scores, best_thr = sc["Overall Accuracy"], sc, thr
                if args.sweep_all:
                    print_row(extractor_label, source, best_scores,
                              note=f"^ best thr={best_thr:.2f}")
                    print()
                else:
                    print_row(extractor_label, source, best_scores,
                              note=f"← best thr={best_thr:.2f}")
                log_entries.append({
                    "system": extractor_label, "source": source,
                    "threshold": best_thr, "best": True, **best_scores,
                })
            else:
                thr = args.thr
                est_f = [apply_threshold(p, c, thr) for p, c in zip(est_p, est_c)] \
                        if est_c is not None else list(est_p)
                sc = evaluate(src_ref_t, src_ref_f, est_t, est_f)
                thr_note = "" if key == "ftanet" else f"thr={thr:.2f}"
                print_row(extractor_label, source, sc, note=thr_note)
                log_entries.append({
                    "system": extractor_label, "source": source,
                    "threshold": None if key == "ftanet" else thr,
                    "best": None, **sc,
                })

        if source != args.sources[-1]:
            print()

    print(f"{sep}\n")

    n_stems_per_source = {src: len(keep) for src, keep in valid_per_source.items()}
    save_results(args, run_scratch_name, run_fine_name, n_stems_per_source, log_entries)


if __name__ == "__main__":
    main()
