"""
Structural embedding PCA plots for svara analysis.

Usage:
    python -m src.utils.plot_embeddings --tag v1_TR
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import polars as pl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import settings as S
from src.analysis.svara_segment_analysis import RECORDINGS, COLORS


def _save(fig, name: str, out_dir: Path, dpi: int = 120) -> None:
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path.name}")


def plot_embeddings(svara_labels, performers, out_dir: Path) -> None:
    """Load structural embeddings and generate PCA plots."""
    n_s = len(svara_labels)
    n_p = len(performers)
    # ── cell 40 ───────────────────────────────────────────────────
    from src.features.structural_embedding import structural_embedding_one_recording

    # Carrega (o recalcula) les embeddings per a les 5 gravacions i les concatena
    dfs = []
    for rec in RECORDINGS:
        df_emb = structural_embedding_one_recording(
            recording_id=rec,
            tonic_hz=S.SARASUDA_TONICS[rec],
            corpus_root=S.DATA_CORPUS,
            interim_root=S.INTERIM_RECORDINGS,
            max_segments=12,
        )
        dfs.append(df_emb)

    df_emb_all = pl.concat(dfs)
    print(f'Total embeddings: {df_emb_all.shape[0]} svaras  ×  {df_emb_all.shape[1]} columnes')
    df_emb_all.head(5)

    # ── cell 41 ───────────────────────────────────────────────────
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # Extreu matriu d'embeddings (llista → array 2D)
    X_raw   = np.array(df_emb_all['embedding'].to_list(), dtype=float)
    labels  = np.array(df_emb_all['svara_label'].to_list())
    rec_ids = np.array(df_emb_all['recording_id'].to_list())
    perfs   = np.array([r.split('_')[2] for r in rec_ids])

    # Filtra files amb algun NaN
    valid = np.all(np.isfinite(X_raw), axis=1)
    X       = X_raw[valid]
    labels  = labels[valid]
    rec_ids = rec_ids[valid]
    perfs   = perfs[valid]

    print(f'Embeddings totals: {X_raw.shape[0]}  |  vàlids (sense NaN): {X.shape[0]}  |  descartats: {(~valid).sum()}')
    print(f'Shape X: {X.shape}')

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    pca    = PCA(n_components=2, random_state=0)
    X_2d   = pca.fit_transform(X_sc)
    print(f'Variància explicada: PC1={pca.explained_variance_ratio_[0]:.1%}  PC2={pca.explained_variance_ratio_[1]:.1%}')

    # ── cell 42 ───────────────────────────────────────────────────
    # Colors per svara_label (consistent amb la raga Saveri: S R G M P D N)
    SVARA_COLORS = {sl: plt.cm.tab10(i / max(len(svara_labels), 1))
                    for i, sl in enumerate(svara_labels)}
    MARKERS = {'bdn': 'o', 'drn': 's', 'psn': '^', 'rkm': 'D', 'svd': 'P'}

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # — Plot A: colorejat per svara_label —
    ax = axes[0]
    for sl in svara_labels:
        mask = np.array([l == sl for l in labels])
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   color=SVARA_COLORS[sl], alpha=0.45, s=18, label=sl, linewidths=0)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.set_title('PCA structural embedding — color per svara')
    ax.legend(title='Svara', fontsize=9, markerscale=1.5)
    ax.grid(alpha=0.2)

    # — Plot B: colorejat per intèrpret —
    ax = axes[1]
    for perf in performers:
        mask = np.array([p == perf for p in perfs])
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   color=COLORS.get(perf, 'gray'), alpha=0.45, s=18,
                   marker=MARKERS.get(perf, 'o'), label=perf, linewidths=0)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.set_title('PCA structural embedding — color per intèrpret')
    ax.legend(title='Intèrpret', fontsize=9, markerscale=1.5)
    ax.grid(alpha=0.2)

    fig.suptitle('Structural embeddings (5 gravacions, totes les svaras) — PCA 2D', fontsize=13)
    plt.tight_layout()
    _save(fig, "fig_42", out_dir)

    # ── cell 44 ───────────────────────────────────────────────────
    # Plot gran: tots els punts colorejats per svara, tots els intèrprets junts
    fig, ax = plt.subplots(figsize=(9, 7))

    for sl in svara_labels:
        mask = labels == sl
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   color=SVARA_COLORS[sl], alpha=0.5, s=22, label=sl, linewidths=0)
        # Centroide
        cx, cy = X_2d[mask, 0].mean(), X_2d[mask, 1].mean()
        ax.scatter(cx, cy, color=SVARA_COLORS[sl], s=180, zorder=6,
                   edgecolors='black', linewidths=1.2, marker='*')
        ax.text(cx, cy + 0.15, sl, ha='center', va='bottom',
                fontsize=11, fontweight='bold', color=SVARA_COLORS[sl])

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=11)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=11)
    ax.set_title('PCA structural embedding — tots els intèrprets, color per svara\n★ = centroide', fontsize=12)
    ax.legend(title='Svara', fontsize=9, markerscale=1.5)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    _save(fig, "13_pca_all", out_dir)

    # ── cell 45 ───────────────────────────────────────────────────
    # Grid: un subplot per svara — fons gris (resta) + color (svara destacada)
    ncols = min(n_s, 4)
    nrows = int(np.ceil(n_s / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows), squeeze=False)

    for idx, sl in enumerate(svara_labels):
        ax  = axes[idx // ncols][idx % ncols]
        mask = labels == sl

        # Fons: tots els altres punts en gris clar
        ax.scatter(X_2d[~mask, 0], X_2d[~mask, 1],
                   color='lightgray', alpha=0.4, s=10, linewidths=0)
        # Svara destacada
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   color=SVARA_COLORS[sl], alpha=0.7, s=22, linewidths=0)
        # Centroide
        cx, cy = X_2d[mask, 0].mean(), X_2d[mask, 1].mean()
        ax.scatter(cx, cy, color=SVARA_COLORS[sl], s=160, zorder=5,
                   edgecolors='black', linewidths=1.2, marker='*')

        ax.set_title(sl, fontsize=14, fontweight='bold', color=SVARA_COLORS[sl])
        ax.set_xlabel(f'PC1', fontsize=9)
        ax.set_ylabel(f'PC2', fontsize=9)
        ax.grid(alpha=0.2)
        ax.text(0.02, 0.97, f'n={mask.sum()}', transform=ax.transAxes,
                fontsize=9, va='top', color='gray')

    # Amaga subplots buits
    for idx in range(n_s, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle('PCA structural embedding per svara — tots els intèrprets combinats  (★ = centroide)', fontsize=13)
    plt.tight_layout()
    _save(fig, "14_pca_grid", out_dir)




def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="v1", help="Output subfolder tag")
    args = parser.parse_args()

    out_dir = S.FIGURES_DIR / "structural_analysis" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[plot_embeddings] tag={args.tag!r}  output → {out_dir}")

    from src.analysis.svara_segment_analysis import load_all
    _, _, svara_labels, performers = load_all()
    plot_embeddings(svara_labels, performers, out_dir)


if __name__ == "__main__":
    main()
