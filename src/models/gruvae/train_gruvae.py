"""
Training script for SvaraGRUVAE.

Run from project root:
    python -m src.models.gruvae.train_gruvae

Edit the CONFIG section below to change hyperparameters.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import random_split, DataLoader

import settings as S
from src.models.gruvae.dataset_gruvae import (
    SvaraDataset,
    collate_svara_batch,
)
from src.models.gruvae.model_gruvae import (
    ModelConfig,
    SvaraGRUVAE,
    DATASET_FEATURE_COLS,
    total_vae_loss,
    linear_kl_beta,
)


# ============================================================
# CONFIG
# ============================================================

RECORDING_IDS  = None    # None → all S.SARASUDA_VARNAM; or e.g. ["srs_v1_bdn_sav"]
EPOCHS         = 100
BATCH_SIZE     = 32
LR             = 5e-4
WARMUP_STEPS   = 8000
VAL_FRACTION   = 0.15
SAVE_EVERY     = 10
SEED           = 42
CHECKPOINT_DIR = Path("data/interim/models/gruvae_v4")

MODEL_CFG = ModelConfig(
    hidden_dim   = 32,   # reduced capacity → decoder can't reconstruct without z
    latent_dim   = 32,
    num_layers   = 1,    # 1 layer: simpler decoder, less able to memorize
    dropout      = 0.0,  # no dropout with 1 layer (PyTorch ignores it anyway)
    max_seq_len  = 15,   # límit per generate(); training usa padding dinàmic
    svara_cond_dim = 7,  # cVAE: condicionat per label de svara (D,G,M,N,P,R,S)
    condition_z_every_step = True,
    teacher_forcing_ratio  = 0.0,   # decoder never sees real prev token → must use z
    lambda_type      = 0.2,
    lambda_dur_cp    = 0.3,   # MI_svara=0.11, MI_perf=0.05 → invariant-ish
    lambda_dur_sta   = 0.05,  # MI_perf=0.26-0.39 → performer-dependent, penalitzar poc
    lambda_dur_sil   = 0.1,   # MI weak
    lambda_cp_cents  = 3.0,
    lambda_sta_cents = 7.0,
    lambda_sil_cents = 0.3,
    lambda_length    = 0.1,
    beta             = 0.5,
    free_bits        = 0.5,  # prevé posterior collapse: KL ≥ 0.5 × 32 = 16 nats
    use_huber_for_continuous = True,
)


# ============================================================
# HELPERS
# ============================================================

class _Tee:
    """Write to stdout and a log file simultaneously."""
    def __init__(self, log_path: Path, append: bool = False):
        mode = "a" if append else "w"
        self._file   = log_path.open(mode, buffering=1)
        self._stdout = sys.stdout
    def write(self, data: str):
        self._stdout.write(data)
        self._file.write(data)
    def flush(self):
        self._stdout.flush()
        self._file.flush()
    def close(self):
        sys.stdout = self._stdout
        self._file.close()


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _make_loaders(
    recording_ids: list[str] | None,
    val_fraction: float,
    batch_size: int,
    max_seq_len: int,
) -> tuple[DataLoader, DataLoader]:
    dataset = SvaraDataset(
        recording_ids=recording_ids,
        feature_cols=DATASET_FEATURE_COLS,
    )
    print(f"  Dataset: {len(dataset)} svaras")

    n_val   = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"  Train: {n_train}  Val: {n_val}")

    # max_len=None → padding dinàmic al màxim del batch
    collate = lambda b: collate_svara_batch(b, max_len=None)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate)
    return train_loader, val_loader


@torch.no_grad()
def _val_epoch(
    model: SvaraGRUVAE,
    loader: DataLoader,
    device: torch.device,
    global_step: int,
) -> dict:
    model.eval()
    totals: dict[str, float] = {}
    n_batches = 0
    for x, lengths, svara_idx in loader:
        x, lengths, svara_idx = x.to(device), lengths.to(device), svara_idx.to(device)
        outputs = model(x, lengths, svara_idx=svara_idx, teacher_forcing_ratio=0.0)
        beta = min(model.cfg.beta, linear_kl_beta(global_step, WARMUP_STEPS))
        _, stats = total_vae_loss(outputs, x, lengths, model.cfg, beta=beta)
        for k, v in stats.items():
            totals[k] = totals.get(k, 0.0) + float(v)
        n_batches += 1
    return {k: v / n_batches for k, v in totals.items()}


def _save_ckpt(path: Path, epoch: int, global_step: int, run_id: int,
               model: SvaraGRUVAE, optimizer, scheduler,
               best_val_loss: float, history: list) -> None:
    torch.save({
        "epoch":         epoch,
        "global_step":   global_step,
        "run_id":        run_id,
        "model":         model.state_dict(),
        "optimizer":     optimizer.state_dict(),
        "scheduler":     scheduler.state_dict(),
        "best_val":      best_val_loss,
        "history":       history,
        "model_cfg":     dataclasses.asdict(model.cfg),
    }, path)


# ============================================================
# MAIN TRAINING LOOP
# ============================================================

def train(
    recording_ids: list[str] | None = RECORDING_IDS,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    warmup_steps: int = WARMUP_STEPS,
    val_fraction: float = VAL_FRACTION,
    checkpoint_dir: Path = CHECKPOINT_DIR,
    model_cfg: ModelConfig = MODEL_CFG,
    new_run: bool = False,
):
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    device = _device()

    train_loader, val_loader = _make_loaders(
        recording_ids, val_fraction, batch_size, model_cfg.max_seq_len
    )

    model     = SvaraGRUVAE(model_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=40,
    )

    # ── run directory ────────────────────────────────────────────────────────
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    existing_runs = sorted(checkpoint_dir.glob("run_*"))

    start_epoch   = 1
    global_step   = 0
    best_val_loss = float("inf")
    history: list[dict] = []
    resumed = False

    if existing_runs and not new_run:
        last_path = existing_runs[-1] / "last.pt"
        if last_path.exists():
            ckpt = torch.load(last_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch   = ckpt["epoch"] + 1
            global_step   = ckpt.get("global_step", 0)
            best_val_loss = ckpt["best_val"]
            history       = ckpt.get("history", [])
            run_id        = ckpt.get("run_id", len(existing_runs))
            resumed       = True

    if not resumed:
        run_id = int(existing_runs[-1].name.split("_")[1]) + 1 if existing_runs else 1

    run_dir = checkpoint_dir / f"run_{run_id:03d}"
    run_dir.mkdir(exist_ok=True)

    tee = _Tee(run_dir / f"run_{run_id:03d}.log", append=resumed)
    sys.stdout = tee

    try:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[train] device={device}  parameters={n_params:,}  run={run_dir}")

        if resumed:
            print(f"[train] Resumed run_{run_id:03d} from epoch {ckpt['epoch']}  "
                  f"(best_val={best_val_loss:.4f})")
            if start_epoch > epochs:
                print(
                    f"[train] nothing to train (checkpoint at {ckpt['epoch']}, requested {epochs}). "
                    f"Pass --epochs {ckpt['epoch'] + 100} to extend, or --new-run to restart."
                )
                return

        for epoch in range(start_epoch, epochs + 1):
            t0 = time.time()

            model.train()
            train_totals: dict[str, float] = {}
            n_batches = 0
            for x, lengths, svara_idx in train_loader:
                x, lengths, svara_idx = x.to(device), lengths.to(device), svara_idx.to(device)
                optimizer.zero_grad()
                beta    = min(model_cfg.beta, linear_kl_beta(global_step, warmup_steps))
                outputs = model(x, lengths, svara_idx=svara_idx)
                loss, stats = total_vae_loss(outputs, x, lengths, model_cfg, beta=beta)
                loss.backward()
                optimizer.step()
                global_step += 1
                for k, v in stats.items():
                    train_totals[k] = train_totals.get(k, 0.0) + float(v)
                n_batches += 1

            train_stats = {k: v / n_batches for k, v in train_totals.items()}
            val_stats   = _val_epoch(model, val_loader, device, global_step)
            val_loss    = float(val_stats["total_loss"])
            elapsed     = time.time() - t0

            val_recon = float(val_stats["recon_loss"])
            is_best   = val_recon < best_val_loss
            if is_best:
                best_val_loss = val_recon

            scheduler.step(val_recon)
            lr_now = optimizer.param_groups[0]["lr"]

            flag = "  <- best" if is_best else ""
            print(
                f"Epoch {epoch:4d}/{epochs}  "
                f"train={train_stats['total_loss']:.4f} (r={train_stats['recon_loss']:.4f})  "
                f"val={val_loss:.4f} (r={val_recon:.4f})  "
                f"kl={train_stats['kl_loss']:.4f}  "
                f"beta={min(model_cfg.beta, linear_kl_beta(global_step, warmup_steps)):.3f}  "
                f"lr={lr_now:.2e}  "
                f"({elapsed:.1f}s){flag}"
            )

            history.append({"epoch": epoch, "train": train_stats, "val": val_stats})

            _save_ckpt(run_dir / "last.pt", epoch, global_step, run_id,
                       model, optimizer, scheduler, best_val_loss, history)
            if is_best:
                _save_ckpt(run_dir / "best.pt", epoch, global_step, run_id,
                           model, optimizer, scheduler, best_val_loss, history)
                _save_ckpt(run_dir / f"epoch_{epoch:06d}_best.pt", epoch, global_step, run_id,
                           model, optimizer, scheduler, best_val_loss, history)
            if epoch % 50 == 0:
                _save_ckpt(run_dir / f"epoch_{epoch:06d}.pt", epoch, global_step, run_id,
                           model, optimizer, scheduler, best_val_loss, history)

            with open(run_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2)

        print(f"\n[train] done. best val={best_val_loss:.4f}  ckpt={run_dir}")

    finally:
        tee.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",   type=int,   default=EPOCHS)
    parser.add_argument("--new-run",  action="store_true",
                        help="Ignora last.pt i comença des de zero")
    args = parser.parse_args()
    train(epochs=args.epochs, new_run=args.new_run)
