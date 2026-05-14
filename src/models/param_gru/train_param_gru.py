"""
Training script for ParamGRU (deterministic conditional GRU regressor).

Run from project root:
    python -m src.models.param_gru.train_param_gru
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
import settings as S

from src.models.param_gru.dataset_param_gru import build_dataset, collate_param_batch
from src.models.param_gru.model_param_gru import (
    ParamGRUConfig, ParamGRU, total_loss, fit_residuals, save_residuals,
    b2_slope_loss,
)


# ============================================================
# CONFIG
# ============================================================

EPOCHS         = 500
BATCH_SIZE     = 32
LR             = 3e-4
VAL_FRACTION   = 0.1
SAVE_EVERY     = 20
SEED           = 42
CHECKPOINT_DIR = S.PARAM_GRU_DIR

MODEL_CFG = ParamGRUConfig(
    hidden_dim   = 64,
    num_layers   = 1,
    cond_dim     = 8,
    lambda_slope = 0.3,
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
        self.f.flush()

    def flush(self) -> None:
        self._orig.flush()
        self.f.flush()


def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


def _run_epoch(
    model: ParamGRU,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool = True,
) -> dict:
    model.train(train)
    totals: dict[str, float] = {}
    n_batches = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = _to_device(batch, device)

            outputs = model(batch)
            loss, stats = total_loss(outputs, batch, model.cfg)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            for k, v in stats.items():
                totals[k] = totals.get(k, 0.0) + float(v)
            n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(CHECKPOINT_DIR.glob("run_*"))
    run_id   = f"run_{len(existing) + 1:03d}"
    run_dir  = CHECKPOINT_DIR / run_id
    run_dir.mkdir()

    sys.stdout = _Tee(run_dir / "log.txt")
    print(f"Run: {run_id}  device={device}  {datetime.datetime.now():%Y-%m-%d %H:%M}")
    print(f"Config: {dataclasses.asdict(MODEL_CFG)}")

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

    model     = ParamGRU(MODEL_CFG).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    n_params  = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,}")

    (run_dir / "config.json").write_text(
        json.dumps(dataclasses.asdict(MODEL_CFG), indent=2)
    )

    best_val = float("inf")
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        tr_stats  = _run_epoch(model, tr_loader,  optimizer, device, train=True)
        val_stats = _run_epoch(model, val_loader, None,      device, train=False)

        val_loss = val_stats["total_loss"]
        elapsed  = time.time() - t0

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"ep={epoch:4d}  "
                f"tr={tr_stats['total_loss']:.4f}  "
                f"val={val_loss:.4f}  "
                f"recon={tr_stats['recon_loss']:.4f}  "
                f"b2={tr_stats['b2_loss']:.4f}  "
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

    # Slope continuity sanity check on val set
    print("\nSlope continuity sanity check (val set)...")
    ckpt_best = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt_best["model"])
    model.eval()
    b2_vals = []
    with torch.no_grad():
        for batch in val_loader:
            batch = _to_device(batch, device)
            out   = model(batch)
            b2    = b2_slope_loss(out["params"], batch["target_mask"],
                                  batch["dur_sec"], batch["delta_cents"])
            b2_vals.append(float(b2))
    print(f"  B2 loss (val): mean={sum(b2_vals)/len(b2_vals):.4f}  "
          f"max={max(b2_vals):.4f}  min={min(b2_vals):.4f}")

    # Fit per-svara residual distributions on training set
    print("\nFitting residual distributions...")
    tr_loader_full = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=False,
                                collate_fn=collate_param_batch)
    ckpt_best = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt_best["model"])
    dists = fit_residuals(model, tr_loader_full, device)
    res_path = run_dir / "residuals.npz"
    save_residuals(dists, res_path)
    print(f"Residual dists: {sorted(dists.keys())}")
    print(f"Saved → {res_path}")


if __name__ == "__main__":
    main()
