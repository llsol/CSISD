"""
Training script for SvaraGRUVAE.

Run from project root:
    python -m src.models.gruvae.train_gruvae

Edit the CONFIG section below to change hyperparameters.
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

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
WARMUP_STEPS   = 8000     # steps until beta reaches 1.0 (longer → slower collapse)
VAL_FRACTION   = 0.15    # fraction of svaras held out for validation
SAVE_EVERY     = 10      # save checkpoint every N epochs
CHECKPOINT_DIR = Path("data/interim/models/gruvae_v4")

MODEL_CFG = ModelConfig(
    hidden_dim   = 32,   # reduced capacity → decoder can't reconstruct without z
    latent_dim   = 32,
    num_layers   = 1,    # 1 layer: simpler decoder, less able to memorize
    dropout      = 0.0,  # no dropout with 1 layer (PyTorch ignores it anyway)
    max_seq_len  = 32,
    condition_z_every_step = True,
    teacher_forcing_ratio  = 0.0,   # decoder never sees real prev token → must use z
    lambda_type      = 0.2,   # type sequence least informative
    lambda_dur       = 1.5,   # CP length important
    lambda_cp_cents  = 10.0,  # CP pitch primary discriminator between classes
    lambda_sta_cents = 1.5,   # STA pitch secondary
    lambda_sil_cents = 0.5,   # SIL pitch inherited, low priority
    lambda_length    = 0.1,
    beta             = 0.3,   # lower KL pressure → encoder can spread mu → visible clusters
    use_huber_for_continuous = True,
)


# ============================================================
# HELPERS
# ============================================================

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

    collate = lambda b: collate_svara_batch(b, max_len=max_seq_len)
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
    for x, lengths in loader:
        x, lengths = x.to(device), lengths.to(device)
        outputs = model(x, lengths, teacher_forcing_ratio=0.0)
        beta = min(model.cfg.beta, linear_kl_beta(global_step, WARMUP_STEPS))
        _, stats = total_vae_loss(outputs, x, lengths, model.cfg, beta=beta)
        for k, v in stats.items():
            totals[k] = totals.get(k, 0.0) + float(v)
        n_batches += 1
    return {k: v / n_batches for k, v in totals.items()}


def _save_checkpoint(
    model: SvaraGRUVAE,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    checkpoint_dir: Path,
    tag: str = "",
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    name = f"checkpoint_epoch{epoch:04d}{tag}.pt"
    path = checkpoint_dir / name
    torch.save({
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_loss":        val_loss,
        "model_cfg":       dataclasses.asdict(model.cfg),
    }, path)
    return path


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
    save_every: int = SAVE_EVERY,
    checkpoint_dir: Path = CHECKPOINT_DIR,
    model_cfg: ModelConfig = MODEL_CFG,
):
    device = _device()
    print(f"Device: {device}")
    print(f"Recordings: {recording_ids or S.SARASUDA_VARNAM}")

    # Data
    train_loader, val_loader = _make_loaders(
        recording_ids, val_fraction, batch_size, model_cfg.max_seq_len
    )

    # Model
    model     = SvaraGRUVAE(model_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    best_val_loss = float("inf")
    global_step   = 0
    log: list[dict] = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # --- train ---
        model.train()
        train_totals: dict[str, float] = {}
        n_batches = 0
        for x, lengths in train_loader:
            x, lengths = x.to(device), lengths.to(device)
            optimizer.zero_grad()
            beta    = min(model_cfg.beta, linear_kl_beta(global_step, warmup_steps))
            outputs = model(x, lengths)
            loss, stats = total_vae_loss(outputs, x, lengths, model_cfg, beta=beta)
            loss.backward()
            optimizer.step()
            global_step += 1
            for k, v in stats.items():
                train_totals[k] = train_totals.get(k, 0.0) + float(v)
            n_batches += 1

        train_stats = {k: v / n_batches for k, v in train_totals.items()}

        # --- val ---
        val_stats = _val_epoch(model, val_loader, device, global_step)
        val_loss  = float(val_stats["total_loss"])

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:4d}/{epochs} | "
            f"train={train_stats['total_loss']:.4f}  "
            f"val={val_loss:.4f}  "
            f"kl={train_stats['kl_loss']:.4f}  "
            f"type={train_stats['type_loss']:.4f}  "
            f"len={train_stats['length_loss']:.4f}  "
            f"beta={min(model_cfg.beta, linear_kl_beta(global_step, warmup_steps)):.3f}  "
            f"({elapsed:.1f}s)"
        )

        log.append({"epoch": epoch, "train": train_stats, "val": val_stats})

        # --- checkpoints ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_dir, tag="_best")

        if epoch % save_every == 0:
            _save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_dir)

    # Save final + training log
    _save_checkpoint(model, optimizer, epochs, val_loss, checkpoint_dir, tag="_final")
    log_path = checkpoint_dir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nDone. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    train()
