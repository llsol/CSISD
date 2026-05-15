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
from src.utils.corpus_stamp import corpus_meta
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

RECORDING_IDS  = None    # None → all S.RECORDING_SELECTION; or e.g. ["srs_v1_bdn_sav"]
CV_CHECKPOINT_DIR = S.GRUVAE_CV_DIR
EPOCHS         = 800
BATCH_SIZE     = 32
LR             = 5e-4
WARMUP_STEPS   = 8000
VAL_FRACTION   = 0.15
SAVE_EVERY     = 10
SEED           = 42
CHECKPOINT_DIR = S.GRUVAE_DIR

# Teacher forcing schedule: linear decay from TF_START to 0 over TF_DECAY epochs
TF_START = 0.5
TF_DECAY = 200   # epoch at which tf_ratio reaches 0

MODEL_CFG = ModelConfig(
    hidden_dim   = 16,
    latent_dim   = 8,
    num_layers   = 1,
    dropout      = 0.0,
    max_seq_len  = 15,
    svara_cond_dim = 7,
    condition_z_every_step = True,
    teacher_forcing_ratio  = 0.0,  # used for val/generate; train uses schedule
    # MI-calibrated lambdas (see svara_mi_analysis.py, 2026-05-13)
    lambda_type       = 0.5,   # grammar = most important structural feature
    lambda_dur_cp     = 0.05,  # MI_sv=0.057 low & performer-contaminated (was 0.3)
    lambda_dur_stap   = 0.1,   # best struct ratio 1.9x, underweighted (was 0.05)
    lambda_dur_stat   = 0.02,  # MI_pf > MI_sv, reduce (was 0.05)
    lambda_dur_sil    = 0.1,   # weak signal, unchanged
    lambda_cp_cents   = 2.0,   # MI_pf=0.126 significant, slight reduction (was 3.0)
    lambda_stap_cents = 7.0,   # ratio 8.9x, well-justified
    lambda_stat_cents = 7.0,   # highest MI_sv=0.840, keep
    lambda_length     = 0.2,   # ratio=17.0x, most performer-invariant (was 0.1)
    lambda_dur_tra    = 0.01,  # MI_sv=0.000, pure performer noise (was 0.1)
    lambda_dur_trd    = 0.02,  # MI_pf > MI_sv (was 0.1)
    beta              = 0.1,
    free_bits         = 0.0,   # floor = 0.1 × 32 = 3.2 nats
    cond_dropout_p    = 0.8,   # 80% seqüències sense cond → decoder ha d'usar z
    cond_in_steps     = False, # cond only in init_hidden, forces grammar through z
    use_hard_mask     = True,  # apply VALID_NEXT mask during training → grammar signal
    use_huber_for_continuous = True,
    use_attention     = True,
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
    val_recording_id: str | None = None,
) -> tuple[DataLoader, DataLoader]:
    collate = lambda b: collate_svara_batch(b, max_len=None)

    if val_recording_id is not None:
        # Leave-one-recording-out split
        all_ids = recording_ids if recording_ids is not None else S.RECORDING_SELECTION
        train_ids = [r for r in all_ids if r != val_recording_id]
        train_ds = SvaraDataset(recording_ids=train_ids,         feature_cols=DATASET_FEATURE_COLS)
        val_ds   = SvaraDataset(recording_ids=[val_recording_id], feature_cols=DATASET_FEATURE_COLS)
        print(f"  Train: {len(train_ds)} svaras ({', '.join(train_ids)})")
        print(f"  Val:   {len(val_ds)} svaras  ({val_recording_id})")
    else:
        # Random split
        dataset = SvaraDataset(recording_ids=recording_ids, feature_cols=DATASET_FEATURE_COLS)
        print(f"  Dataset: {len(dataset)} svaras")
        n_val   = max(1, int(len(dataset) * val_fraction))
        n_train = len(dataset) - n_val
        train_ds, val_ds = random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )
        print(f"  Train: {n_train}  Val: {n_val}")

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
        x, lengths = x.to(device), lengths.to(device)
        svara_idx = svara_idx.to(device)
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
    fold: int | None = None,
):
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    device = _device()

    # Fold overrides: leave-one-recording-out + dedicated checkpoint dir
    val_recording_id: str | None = None
    if fold is not None:
        val_recording_id = S.RECORDING_SELECTION[fold]
        short = val_recording_id.split("_")[2]   # e.g. "bdn"
        checkpoint_dir = CV_CHECKPOINT_DIR / f"fold_{fold}_{short}"
        print(f"[train] 5-fold CV  fold={fold}  val={val_recording_id}")

    train_loader, val_loader = _make_loaders(
        recording_ids, val_fraction, batch_size, model_cfg.max_seq_len,
        val_recording_id=val_recording_id,
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

    if not resumed:
        (run_dir / "config.json").write_text(
            json.dumps({**dataclasses.asdict(model_cfg), **corpus_meta()}, indent=2)
        )

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
            tf_ratio = max(0.0, TF_START * (1.0 - epoch / TF_DECAY))
            train_totals: dict[str, float] = {}
            n_batches = 0
            for x, lengths, svara_idx in train_loader:
                x, lengths = x.to(device), lengths.to(device)
                svara_idx = svara_idx.to(device)
                optimizer.zero_grad()
                beta    = min(model_cfg.beta, linear_kl_beta(global_step, warmup_steps))
                outputs = model(x, lengths, svara_idx=svara_idx,
                                teacher_forcing_ratio=tf_ratio)
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
            beta_now = min(model_cfg.beta, linear_kl_beta(global_step, warmup_steps))
            def _fmt(s, key): return f"{s[key]:.4f}"
            print(
                f"Epoch {epoch:4d}/{epochs}  "
                f"train={_fmt(train_stats,'total_loss')} "
                f"(type={_fmt(train_stats,'type_loss')} "
                f"dur={_fmt(train_stats,'dur_loss')} "
                f"cents={_fmt(train_stats,'cents_loss')} "
                f"len={_fmt(train_stats,'length_loss')} "
                f"kl={_fmt(train_stats,'kl_loss')})  "
                f"val={val_loss:.4f} "
                f"(type={_fmt(val_stats,'type_loss')} "
                f"dur={_fmt(val_stats,'dur_loss')} "
                f"cents={_fmt(val_stats,'cents_loss')} "
                f"len={_fmt(val_stats,'length_loss')} "
                f"kl={_fmt(val_stats,'kl_loss')})  "
                f"beta={beta_now:.3f}  tf={tf_ratio:.2f}  lr={lr_now:.2e}  ({elapsed:.1f}s){flag}"
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
    parser.add_argument("--fold",     type=int,   default=None, choices=[0, 1, 2, 3, 4],
                        help="Leave-one-out fold (0-4). Activa 5-fold CV.")
    parser.add_argument("--all-folds", action="store_true",
                        help="Entrena els 5 folds seqüencialment")
    args = parser.parse_args()
    if args.all_folds:
        for f in range(5):
            print(f"\n{'='*60}\n  FOLD {f}/4\n{'='*60}")
            train(epochs=args.epochs, new_run=args.new_run, fold=f)
    else:
        train(epochs=args.epochs, new_run=args.new_run, fold=args.fold)
