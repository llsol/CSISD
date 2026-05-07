"""
Latent-space analysis of SvaraGRUVAE to discover svara-forms.

Pipeline:
    1. Encode all svaras in the dataset → collect (z, mu, svara_label).
    2. t-SNE on the full z matrix → global overview plot.
    3. For each svara label: k-means with k=2..K_MAX, pick best k by
       silhouette score → these clusters are the discovered svara-forms.
    4. Per-svara t-SNE plot coloured by form.
    5. Silhouette summary plot.
    6. Save latent vectors + form assignments to disk.

Outputs:
    figures/gruvae/latent_analysis/
        tsne_all_svaras.png
        silhouette_scores.png
        svara_forms/
            {S,R,G,M,P,D,N}_forms.png

    data/interim/gruvae_latent/
        latent_z.npz          z, mu, svara_label (str), sample_id (int)
        svara_forms.json      {svara: {k: int, silhouette: float,
                                       labels: [int, ...]}}

Usage:
    python -m src.models.gruvae.latent_analysis
    python -m src.models.gruvae.latent_analysis --run run_001 --k-max 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import settings
from src.models.gruvae.dataset_gruvae import (
    SvaraDataset,
    SVARA_LABELS,
    SVARA_TO_IDX,
    collate_svara_batch,
)
from src.models.gruvae.model_gruvae import ModelConfig, SvaraGRUVAE, DATASET_FEATURE_COLS

CKPT_DIR    = settings.DATA_INTERIM / "models" / "gruvae_v4"
OUT_FIGS    = settings.FIGURES_DIR  / "gruvae" / "latent_analysis"
OUT_DATA    = settings.DATA_INTERIM / "gruvae_latent"
IDX_TO_SVARA = {v: k for k, v in SVARA_TO_IDX.items()}

SVARA_COLORS = {
    "D": "#e6194b", "G": "#3cb44b", "M": "#4363d8",
    "N": "#f58231", "P": "#911eb4", "R": "#42d4f4", "S": "#f032e6",
}
K_MAX_DEFAULT = 8
TSNE_PERP     = 30
SEED          = 42


# ── model ─────────────────────────────────────────────────────────────────────

def load_model(run: str | None, device: torch.device) -> SvaraGRUVAE:
    run_dir = (CKPT_DIR / run) if run else sorted(CKPT_DIR.glob("run_*"))[-1]
    best    = run_dir / "best.pt"
    if not best.exists():
        best = run_dir / "last.pt"
    ckpt  = torch.load(best, map_location=device)
    cfg   = ModelConfig(**ckpt["model_cfg"])
    model = SvaraGRUVAE(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[latent] loaded  epoch={ckpt['epoch']}  best_val={ckpt['best_val']:.4f}  {best}")
    return model


# ── encoding ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_dataset(model: SvaraGRUVAE, device: torch.device) -> dict:
    dataset = SvaraDataset(recording_ids=None, feature_cols=DATASET_FEATURE_COLS)
    loader  = DataLoader(
        dataset, batch_size=64, shuffle=False,
        collate_fn=lambda b: collate_svara_batch(b, max_len=None),
    )
    all_mu, all_svara_idx = [], []

    for x, lengths, svara_idx in loader:
        x         = x.to(device)
        lengths   = lengths.to(device)
        svara_idx = svara_idx.to(device)
        mu, _     = model.encoder(x, lengths)
        all_mu.append(mu.cpu().numpy())
        all_svara_idx.append(svara_idx.cpu().numpy())

    mu          = np.concatenate(all_mu,        axis=0)   # (N, latent_dim)
    svara_idx   = np.concatenate(all_svara_idx, axis=0)   # (N,)
    svara_label = np.array([IDX_TO_SVARA[i] for i in svara_idx])
    rec_id      = np.array(dataset._recording_ids)

    print(f"[latent] encoded {len(mu)} svaras  latent_dim={mu.shape[1]}")
    return {"mu": mu, "svara_idx": svara_idx, "svara_label": svara_label, "rec_id": rec_id}


# ── t-SNE ─────────────────────────────────────────────────────────────────────

def compute_tsne(z: np.ndarray) -> np.ndarray:
    print(f"[latent] t-SNE on {len(z)} points …", end=" ", flush=True)
    emb = TSNE(n_components=2, perplexity=TSNE_PERP, random_state=SEED,
               init="pca", learning_rate="auto").fit_transform(z)
    print("done")
    return emb


# ── clustering ────────────────────────────────────────────────────────────────

def best_kmeans(z_svara: np.ndarray, k_max: int) -> tuple[int, float, np.ndarray]:
    best_k, best_sil, best_labels = 2, -1.0, np.zeros(len(z_svara), dtype=int)
    for k in range(2, min(k_max + 1, len(z_svara))):
        labels = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit_predict(z_svara)
        if len(np.unique(labels)) < 2:
            continue
        sil = silhouette_score(z_svara, labels)
        if sil > best_sil:
            best_k, best_sil, best_labels = k, sil, labels
    return best_k, best_sil, best_labels


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_tsne_all(emb: np.ndarray, svara_label: np.ndarray, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    for sv in SVARA_LABELS:
        mask = svara_label == sv
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=SVARA_COLORS[sv], label=sv, s=12, alpha=0.6)
    ax.legend(title="Svara", markerscale=2, fontsize=9)
    ax.set_title("t-SNE μ — colored by svara", fontsize=11)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.axis("equal"); ax.tick_params(left=False, bottom=False,
                                     labelleft=False, labelbottom=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[latent] → {out}")


def plot_tsne_by_performer(emb: np.ndarray, rec_id: np.ndarray, out: Path) -> None:
    unique_recs = sorted(set(rec_id))
    cmap        = plt.colormaps.get_cmap("tab10")
    fig, ax     = plt.subplots(figsize=(8, 7))
    for i, rid in enumerate(unique_recs):
        mask  = rec_id == rid
        label = rid.split("_")[2]          # e.g. "bdn"
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=[cmap(i)], label=label, s=12, alpha=0.6)
    ax.legend(title="Performer", markerscale=2, fontsize=9)
    ax.set_title("t-SNE μ — colored by performer/recording", fontsize=11)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.axis("equal"); ax.tick_params(left=False, bottom=False,
                                     labelleft=False, labelbottom=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[latent] → {out}")


def plot_svara_forms(
    emb_sv: np.ndarray,
    labels:  np.ndarray,
    svara:   str,
    k:       int,
    sil:     float,
    out:     Path,
) -> None:
    cmap   = plt.cm.get_cmap("tab10", k)
    fig, ax = plt.subplots(figsize=(6, 5))
    for form_id in range(k):
        mask = labels == form_id
        ax.scatter(emb_sv[mask, 0], emb_sv[mask, 1],
                   c=[cmap(form_id)], label=f"form {form_id}",
                   s=20, alpha=0.75)
    ax.legend(markerscale=1.5, fontsize=8, loc="best")
    ax.set_title(f"Svara {svara} — {k} forms  (silhouette={sil:.3f})", fontsize=10)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.axis("equal"); ax.tick_params(left=False, bottom=False,
                                     labelleft=False, labelbottom=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[latent] → {out}")


def plot_silhouettes(results: dict, out: Path) -> None:
    svaras = SVARA_LABELS
    ks     = [results[s]["k"]         for s in svaras]
    sils   = [results[s]["silhouette"] for s in svaras]
    ns     = [results[s]["n_samples"]  for s in svaras]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(svaras, sils,
                  color=[SVARA_COLORS[s] for s in svaras], alpha=0.8)
    for bar, k, n in zip(bars, ks, ns):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"k={k}\nn={n}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Silhouette score")
    ax.set_title("Best k-means silhouette per svara (svara-form discovery)")
    ax.set_ylim(0, max(sils) * 1.25)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[latent] → {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Latent-space analysis of SvaraGRUVAE for svara-form discovery."
    )
    parser.add_argument("--run",    default=None, help="Run dir (default: latest)")
    parser.add_argument("--k-max",  type=int, default=K_MAX_DEFAULT,
                        help="Max k for k-means search")
    args = parser.parse_args()

    OUT_FIGS.mkdir(parents=True, exist_ok=True)
    (OUT_FIGS / "svara_forms").mkdir(exist_ok=True)
    OUT_DATA.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(args.run, device)

    # ── encode ──────────────────────────────────────────────────────────────
    enc = encode_dataset(model, device)
    mu, svara_label, rec_id = enc["mu"], enc["svara_label"], enc["rec_id"]

    np.savez_compressed(
        OUT_DATA / "latent_z.npz",
        mu=mu,
        svara_idx=enc["svara_idx"],
        svara_label=svara_label.astype(str),
        rec_id=rec_id.astype(str),
    )
    print(f"[latent] saved latent vectors → {OUT_DATA / 'latent_z.npz'}")

    # ── global t-SNE (on mu, not z) ─────────────────────────────────────────
    emb = compute_tsne(mu)
    plot_tsne_all(emb, svara_label, OUT_FIGS / "tsne_by_svara.png")
    plot_tsne_by_performer(emb, rec_id, OUT_FIGS / "tsne_by_performer.png")

    # ── per-svara clustering ─────────────────────────────────────────────────
    results = {}
    for sv in SVARA_LABELS:
        mask   = svara_label == sv
        z_sv   = mu[mask]
        emb_sv = emb[mask]
        n      = len(z_sv)
        print(f"[latent] {sv}: {n} samples → k-means …")

        if n < 4:
            print(f"  [skip] too few samples")
            results[sv] = {"k": 1, "silhouette": 0.0, "n_samples": n, "labels": [0] * n}
            continue

        k, sil, labels = best_kmeans(z_sv, args.k_max)
        results[sv] = {
            "k":          k,
            "silhouette": float(sil),
            "n_samples":  n,
            "labels":     labels.tolist(),
        }
        print(f"  best k={k}  silhouette={sil:.3f}")
        plot_svara_forms(emb_sv, labels, sv, k, sil,
                         OUT_FIGS / "svara_forms" / f"{sv}_forms.png")

    plot_silhouettes(results, OUT_FIGS / "silhouette_scores.png")

    with open(OUT_DATA / "svara_forms.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[latent] saved form assignments → {OUT_DATA / 'svara_forms.json'}")

    # ── summary ──────────────────────────────────────────────────────────────
    print("\n── Svara-form summary ──────────────────────────────────────")
    print(f"  {'Svara':<6} {'n':>5}  {'k':>3}  {'silhouette':>10}")
    for sv in SVARA_LABELS:
        r = results[sv]
        print(f"  {sv:<6} {r['n_samples']:>5}  {r['k']:>3}  {r['silhouette']:>10.3f}")
    print("────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
