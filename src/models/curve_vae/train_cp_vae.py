"""
Train the CP convolutional VAE.

Usage:
    python -m src.models.curve_vae.train_cp_vae --new-run
    python -m src.models.curve_vae.train_cp_vae           # continue last run
    python -m src.models.curve_vae.train_cp_vae --epochs 500
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import settings as S
from src.models.gruvae.dataset_gruvae import SVARA_TO_IDX
from src.models.curve_vae.cp_vae import CPVAE, CPVAEConfig, L_CANONICAL

DATA_PATH = S.DATA_INTERIM / "models" / "curve_vae" / "gt_cp_curves.parquet"
CKPT_DIR  = S.DATA_INTERIM / "models" / "curve_vae" / "cp_vae_runs"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CPDataset(Dataset):
    def __init__(self, df: pl.DataFrame) -> None:
        curves     = np.array(df["curve"].to_list(), dtype=np.float32)  # (N, 64)
        # normalise: divide by corpus-wide std so values ~ N(0,1)
        self.scale = float(curves.std()) + 1e-6
        self.curves    = torch.from_numpy(curves / self.scale).unsqueeze(1)  # (N,1,64)
        self.svara_idx = torch.tensor(
            [SVARA_TO_IDX.get(sv, 0) for sv in df["svara_label"].to_list()],
            dtype=torch.long,
        )
        self.dur_sec   = torch.tensor(df["dur_sec"].to_list(), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.curves)

    def __getitem__(self, i: int):
        return self.curves[i], self.svara_idx[i], self.dur_sec[i]


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _latest_run(ckpt_dir: Path) -> Path | None:
    runs = sorted(ckpt_dir.glob("run_*"))
    return runs[-1] if runs else None


def _new_run(ckpt_dir: Path) -> Path:
    existing = sorted(ckpt_dir.glob("run_*"))
    idx      = len(existing) + 1
    run_dir  = ckpt_dir / f"run_{idx:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def train_epoch(model, loader, opt, device) -> dict:
    model.train()
    totals = {"loss": 0.0, "mse": 0.0, "kl": 0.0}
    for x, sv, dur in loader:
        x, sv, dur = x.to(device), sv.to(device), dur.to(device)
        out = model(x, sv, dur)
        ls  = model.loss(x, out)
        opt.zero_grad()
        ls["loss"].backward()
        opt.step()
        for k in totals:
            totals[k] += float(ls[k])
    n = len(loader)
    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def val_epoch(model, loader, device) -> dict:
    model.eval()
    totals = {"loss": 0.0, "mse": 0.0, "kl": 0.0}
    for x, sv, dur in loader:
        x, sv, dur = x.to(device), sv.to(device), dur.to(device)
        out = model(x, sv, dur)
        ls  = model.loss(x, out)
        for k in totals:
            totals[k] += float(ls[k])
    n = len(loader)
    return {k: v / n for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-run",  action="store_true")
    parser.add_argument("--epochs",   type=int,   default=300)
    parser.add_argument("--batch",    type=int,   default=32)
    parser.add_argument("--lr",       type=float, default=3e-4)
    parser.add_argument("--latent",   type=int,   default=4)
    parser.add_argument("--val-frac", type=float, default=0.15)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df      = pl.read_parquet(DATA_PATH)
    dataset = CPDataset(df)
    print(f"Corpus: {len(dataset)} CP curves  scale={dataset.scale:.3f}¢")

    n_val   = max(1, int(len(dataset) * args.val_frac))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, drop_last=False)
    print(f"Train: {n_train}  Val: {n_val}")

    # ── checkpoint setup ──────────────────────────────────────────────
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    best_val    = float("inf")

    if args.new_run or _latest_run(CKPT_DIR) is None:
        run_dir = _new_run(CKPT_DIR)
        cfg     = CPVAEConfig(latent_dim=args.latent)
        model   = CPVAE(cfg).to(device)
        print(f"New run: {run_dir.name}")
    else:
        run_dir = _latest_run(CKPT_DIR)
        ckpt    = torch.load(run_dir / "last.pt", map_location=device)
        cfg     = CPVAEConfig(**ckpt["cfg"])
        model   = CPVAE(cfg).to(device)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt["epoch"]
        best_val    = ckpt["best_val"]
        args.epochs = max(args.epochs, start_epoch + 1)
        print(f"Resume {run_dir.name}  epoch={start_epoch}  best_val={best_val:.6f}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # ── training loop ─────────────────────────────────────────────────
    for epoch in range(start_epoch + 1, args.epochs + 1):
        t0    = time.time()
        tr    = train_epoch(model, train_loader, opt, device)
        vl    = val_epoch(model,   val_loader,         device)
        elapsed = time.time() - t0

        is_best = vl["loss"] < best_val
        if is_best:
            best_val = vl["loss"]

        ckpt_data = {
            "epoch":    epoch,
            "model":    model.state_dict(),
            "cfg":      {"latent_dim": cfg.latent_dim,
                         "channels":   cfg.channels,
                         "beta":       cfg.beta,
                         "free_bits":  cfg.free_bits},
            "best_val": best_val,
            "scale":    dataset.scale,
        }
        torch.save(ckpt_data, run_dir / "last.pt")
        if is_best:
            torch.save(ckpt_data, run_dir / "best.pt")

        if epoch % 10 == 0 or epoch <= 5 or is_best:
            flag = " ★" if is_best else ""
            print(
                f"Ep {epoch:>4}/{args.epochs}  "
                f"tr_loss={tr['loss']:.4f}  tr_mse={tr['mse']:.4f}  tr_kl={tr['kl']:.4f}  "
                f"val_loss={vl['loss']:.4f}  val_mse={vl['mse']:.4f}  "
                f"best={best_val:.4f}  {elapsed:.1f}s{flag}"
            )

    print(f"\nDone. Best val_loss={best_val:.4f}  ckpt → {run_dir}")


if __name__ == "__main__":
    main()
