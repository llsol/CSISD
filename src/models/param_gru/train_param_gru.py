"""
Training script for ParamGRUVAE.

Run from project root:
    python -m src.models.param_gru.train_param_gru

Edit the CONFIG section to change hyperparameters.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.models.param_gru.dataset_param_gru import build_dataset, collate_param_batch
from src.models.param_gru.model_param_gru import (
    ParamGRUConfig, ParamGRUVAE, total_loss,
)
from src.models.gruvae.model_gruvae import linear_kl_beta


# ============================================================
# CONFIG
# ============================================================

EPOCHS         = 500
BATCH_SIZE     = 32
LR             = 3e-4
WARMUP_STEPS   = 3000
VAL_FRACTION   = 0.1
SAVE_EVERY     = 20
SEED           = 42
CHECKPOINT_DIR = Path("data/interim/models/param_gru")

MODEL_CFG = ParamGRUConfig(
    hidden_dim    = 64,
    latent_dim    = 16,
    num_layers    = 1,
    cond_dim      = 8,
    use_attention = True,
    beta          = 0.3,
    free_bits     = 0.5,
    lambda_slope  = 0.05,
)


# ============================================================
# HELPERS
# ============================================================

class _Tee:
    def __init__(self, path: Path):
        self.f = path.open("w")
        self._orig = sys.stdout

    def write(self, s: str) -> None:
        self._orig.write(s)
        self.f.write(s)

    def flush(self) -> None:
        self._orig.flush()
        self.f.flush()


def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


def _run_epoch(
    model: ParamGRUVAE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    global_step: int,
    warmup_steps: int,
    device: torch.device,
    train: bool = True,
) -> tuple[dict, int]:
    model.train(train)
    totals: dict[str, float] = {}
    n_batches = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = _to_device(batch, device)
            beta  = linear_kl_beta(global_step, warmup_steps) if train else 1.0

            outputs = model(batch)
            loss, stats = total_loss(outputs, batch, model.cfg, beta=beta)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                global_step += 1

            for k, v in stats.items():
                totals[k] = totals.get(k, 0.0) + float(v)
            n_batches += 1

    means = {k: v / max(n_batches, 1) for k, v in totals.items()}
    return means, global_step


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── run dir ──────────────────────────────────────────────
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(CHECKPOINT_DIR.glob("run_*"))
    run_id   = f"run_{len(existing) + 1:03d}"
    run_dir  = CHECKPOINT_DIR / run_id
    run_dir.mkdir()

    sys.stdout = _Tee(run_dir / "log.txt")
    print(f"Run: {run_id}  device={device}  {datetime.datetime.now():%Y-%m-%d %H:%M}")
    print(f"Config: {dataclasses.asdict(MODEL_CFG)}")

    # ── data ─────────────────────────────────────────────────
    dataset = build_dataset()
    print(f"Dataset: {len(dataset)} sequences")

    n_val  = max(1, int(len(dataset) * VAL_FRACTION))
    n_tr   = len(dataset) - n_val
    tr_ds, val_ds = random_split(
        dataset, [n_tr, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )
    tr_loader  = DataLoader(tr_ds,  batch_size=BATCH_SIZE, shuffle=True,
                            collate_fn=collate_param_batch, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_param_batch)
    print(f"Train: {n_tr}  Val: {n_val}")

    # ── model ─────────────────────────────────────────────────
    model     = ParamGRUVAE(MODEL_CFG).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    n_params  = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,}")

    # save config
    (run_dir / "config.json").write_text(
        json.dumps(dataclasses.asdict(MODEL_CFG), indent=2)
    )

    # ── training loop ─────────────────────────────────────────
    best_val  = float("inf")
    global_step = 0
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        tr_stats, global_step = _run_epoch(
            model, tr_loader, optimizer, global_step, WARMUP_STEPS, device, train=True
        )
        val_stats, _ = _run_epoch(
            model, val_loader, None, global_step, WARMUP_STEPS, device, train=False
        )

        val_loss = val_stats["total_loss"]
        elapsed  = time.time() - t0

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"ep={epoch:4d}  "
                f"tr={tr_stats['total_loss']:.4f}  "
                f"val={val_loss:.4f}  "
                f"recon={tr_stats['recon_loss']:.4f}  "
                f"kl={tr_stats['kl_loss']:.4f}  "
                f"b2={tr_stats['b2_loss']:.4f}  "
                f"beta={tr_stats['beta']:.3f}  "
                f"t={elapsed:.0f}s"
            )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "cfg": dataclasses.asdict(MODEL_CFG)},
                run_dir / "best.pt",
            )

        if epoch % SAVE_EVERY == 0:
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "cfg": dataclasses.asdict(MODEL_CFG)},
                run_dir / f"ep{epoch:04d}.pt",
            )

    print(f"\nDone. Best val loss: {best_val:.4f}")
    print(f"Saved → {run_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
