"""
Train SwiftF0-scratch from scratch on the SCMS Carnatic dataset.

Architecture differences vs swiftf0_finetune:
    N_FFT=2048 (31 ¢/bin STFT), 360 pitch bins (18 ¢/bin), no pretrained weights.

Loss:
    voicing BCE  — binary cross-entropy on voiced/unvoiced decision
    pitch CE     — cross-entropy over 360 pitch bins (voiced frames only)
    total = voicing_weight * voicing_bce + pitch_ce

Usage:
    python -m src.pitch_extraction.swiftf0_scratch.train
    python -m src.pitch_extraction.swiftf0_scratch.train --new-run

To extend a completed run with more epochs:
    python -m src.pitch_extraction.swiftf0_scratch.train --epochs 150

Checkpoints:
    data/interim/models/swiftf0_scratch/run_001/epoch_001.pt
    data/interim/models/swiftf0_scratch/run_001/last.pt
    data/interim/models/swiftf0_scratch/run_001/best.pt
    data/interim/models/swiftf0_scratch/run_001/history.json
    data/interim/models/swiftf0_scratch/swiftf0_scratch.onnx
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import settings
from src.pitch_extraction.swiftf0_scratch.model import SwiftF0Scratch, N_PITCH_BINS
from src.pitch_extraction.swiftf0_finetune.dataset import SCMSDataset, collate_fn, scms_official_split

# ── hyper-parameters ─────────────────────────────────────────────────────────
BATCH_SIZE     = 4
EPOCHS         = 100
LR             = 1e-4
VOICING_WEIGHT = 1.0
SEED           = 42
# ─────────────────────────────────────────────────────────────────────────────

CKPT_DIR = settings.DATA_INTERIM / "models" / "swiftf0_scratch"


def hz_to_bin(f0_hz: torch.Tensor, bin_centers: torch.Tensor) -> torch.Tensor:
    """f0_hz (B, T) → nearest bin index (B, T), log-space distance."""
    log_f0      = torch.log(f0_hz.unsqueeze(-1).clamp(min=1.0))   # (B, T, 1)
    log_centers = torch.log(bin_centers.clamp(min=1.0))            # (360,)
    return (log_f0 - log_centers).abs().argmin(dim=-1)             # (B, T)


def pitch_cross_entropy(
    pitch_probs: torch.Tensor,   # (B, 360, T)
    f0_hz:       torch.Tensor,   # (B, T)
    voiced:      torch.Tensor,   # (B, T) bool
    bin_centers: torch.Tensor,   # (360,)
) -> torch.Tensor:
    if not voiced.any():
        return pitch_probs.sum() * 0.0

    target = hz_to_bin(f0_hz, bin_centers)

    B, _, T = pitch_probs.shape
    probs_flat  = pitch_probs.permute(0, 2, 1).reshape(B * T, N_PITCH_BINS)
    target_flat = target.reshape(B * T)
    voiced_flat = voiced.reshape(B * T)

    return nn.functional.cross_entropy(probs_flat[voiced_flat], target_flat[voiced_flat])


def run_epoch(
    model:     SwiftF0Scratch,
    loader:    DataLoader,
    optimizer: optim.Optimizer | None,
    device:    torch.device,
    train:     bool,
) -> dict[str, float]:
    model.train(train)
    total_loss = pitch_loss_sum = voicing_loss_sum = 0.0
    n_batches  = 0

    for batch in loader:
        audio  = batch["audio"].to(device)
        f0     = batch["f0"].to(device)
        voiced = batch["voiced"].to(device)

        with torch.set_grad_enabled(train):
            _, confidence, pitch_probs = model(audio)

            T_model = confidence.shape[1]
            T_label = voiced.shape[1]
            T       = min(T_model, T_label)
            confidence  = confidence[:, :T]
            pitch_probs = pitch_probs[:, :, :T]
            f0          = f0[:, :T]
            voiced      = voiced[:, :T]

            v_loss = nn.functional.binary_cross_entropy(
                confidence.clamp(1e-6, 1 - 1e-6), voiced.float()
            )
            p_loss = pitch_cross_entropy(
                pitch_probs, f0, voiced, model.pitch_bin_centers
            )
            loss = VOICING_WEIGHT * v_loss + p_loss

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss       += loss.item()
        pitch_loss_sum   += p_loss.item()
        voicing_loss_sum += v_loss.item()
        n_batches        += 1

    n = max(n_batches, 1)
    return {
        "loss":    total_loss       / n,
        "pitch":   pitch_loss_sum   / n,
        "voicing": voicing_loss_sum / n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scms-root",  default=None)
    parser.add_argument("--epochs",     type=int,   default=EPOCHS)
    parser.add_argument("--batch-size", type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=LR)
    parser.add_argument("--new-run",    action="store_true",
                        help="Force a new run directory instead of resuming the last one")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}")

    scms_root = Path(args.scms_root) if args.scms_root else \
                settings.PROJECT_ROOT / "data" / "datasets" / "scms"

    train_stems, test_stems = scms_official_split(scms_root)
    train_ds = SCMSDataset(scms_root, split=train_stems, crop_sec=0.0)
    val_ds   = SCMSDataset(scms_root, split=test_stems,  crop_sec=0.0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  collate_fn=collate_fn, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_fn, num_workers=2)
    print(f"[train] train={len(train_ds)}  val={len(val_ds)}")

    model = SwiftF0Scratch().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] Parameters: {n_params:,}  (training from scratch)")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )

    # ── run directory ────────────────────────────────────────────────────────
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    existing_runs = sorted(CKPT_DIR.glob("run_*"))
    start_epoch = 1
    best_val    = float("inf")
    history     = []
    resumed     = False

    if existing_runs and not args.new_run:
        last_path = existing_runs[-1] / "last.pt"
        if last_path.exists():
            ckpt = torch.load(last_path, map_location=device)
            if ckpt["epoch"] < args.epochs:
                model.load_state_dict(ckpt["model"])
                optimizer.load_state_dict(ckpt["optimizer"])
                scheduler.load_state_dict(ckpt["scheduler"])
                best_val    = ckpt["best_val"]
                start_epoch = ckpt["epoch"] + 1
                history     = ckpt.get("history", [])
                run_id      = ckpt["run_id"]
                resumed     = True
                print(f"[train] Resumed run_{run_id:03d} from epoch {ckpt['epoch']}  "
                      f"(best_val={best_val:.4f})")
            else:
                print(f"[train] Run {existing_runs[-1].name} complete — starting new run.")

    if not resumed:
        run_id = len(existing_runs) + 1

    run_dir = CKPT_DIR / f"run_{run_id:03d}"
    run_dir.mkdir(exist_ok=True)
    print(f"[train] Run dir: {run_dir}")

    # ── training loop ────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        t0            = time.time()
        train_metrics = run_epoch(model, train_loader, optimizer, device, train=True)
        val_metrics   = run_epoch(model, val_loader,   None,      device, train=False)
        elapsed       = time.time() - t0
        lr_now        = optimizer.param_groups[0]["lr"]

        scheduler.step(val_metrics["loss"])

        is_best = val_metrics["loss"] < best_val
        if is_best:
            best_val = val_metrics["loss"]

        record = {
            "epoch":   epoch,
            "run_id":  run_id,
            "lr":      lr_now,
            "elapsed": round(elapsed, 1),
            "train":   train_metrics,
            "val":     val_metrics,
        }
        history.append(record)

        flag = "  <- best" if is_best else ""
        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train={train_metrics['loss']:.4f} "
            f"(p={train_metrics['pitch']:.4f} v={train_metrics['voicing']:.4f})  "
            f"val={val_metrics['loss']:.4f} "
            f"(p={val_metrics['pitch']:.4f} v={val_metrics['voicing']:.4f})  "
            f"lr={lr_now:.1e}  {elapsed:.0f}s{flag}"
        )

        ckpt_data = {
            "epoch":     epoch,
            "run_id":    run_id,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val":  best_val,
            "history":   history,
        }

        torch.save(ckpt_data, run_dir / f"epoch_{epoch:03d}.pt")
        torch.save(ckpt_data, run_dir / "last.pt")
        if is_best:
            torch.save(ckpt_data, run_dir / "best.pt")

        with open(run_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    # ── export ───────────────────────────────────────────────────────────────
    best_ckpt = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model"])
    model.export_onnx(CKPT_DIR / "swiftf0_scratch.onnx")
    print(f"\n[train] Done. Best val loss: {best_val:.4f}")
    print(f"[train] Run dir: {run_dir}")


if __name__ == "__main__":
    main()
