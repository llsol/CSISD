"""
Svara structural analysis — 5 peces Sarasuda.

Entry point que orquestra:
  - src.analysis.svara_segment_analysis  (càrrega de dades)
  - src.utils.plot_segment_stats         (plots distribució segments)
  - src.utils.plot_embeddings            (PCA embedding)
  - src.analysis.svara_mi_analysis       (MI + invariança + AUROC)

Usage:
    python -m src.utils.svara_structural_analysis --tag v1_TR
    python -m src.utils.svara_structural_analysis --tag v2 --skip-mi
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import settings as S
from src.analysis.svara_segment_analysis import load_all
from src.utils.plot_segment_stats import plot_segment_stats
from src.utils.plot_embeddings import plot_embeddings
from src.analysis.svara_mi_analysis import run_mi_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Svara structural analysis — 5 peces")
    parser.add_argument("--tag", default="v1",
                        help="Subfolder tag (figures/structural_analysis/{tag}/)")
    parser.add_argument("--skip-mi", action="store_true",
                        help="Skip MI/invariance analysis (faster)")
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="Skip PCA embedding plots")
    args = parser.parse_args()

    out_dir = S.FIGURES_DIR / "structural_analysis" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[structural_analysis] tag={args.tag!r}  output → {out_dir}")

    print("[1/4] Loading data...")
    all_rows, df, svara_labels, performers = load_all()
    print(f"      {df.shape[0]} svaras  |  {svara_labels}  |  {performers}")

    print("[2/4] Segment distribution plots...")
    plot_segment_stats(all_rows, df, svara_labels, performers, out_dir)

    if not args.skip_embeddings:
        print("[3/4] Embedding PCA plots...")
        plot_embeddings(svara_labels, performers, out_dir)
    else:
        print("[3/4] Skipped embeddings.")

    if not args.skip_mi:
        print("[4/4] MI + invariance analysis...")
        run_mi_analysis(df, svara_labels, performers, out_dir)
    else:
        print("[4/4] Skipped MI analysis.")

    print(f"\nDone. → {out_dir}")


if __name__ == "__main__":
    main()
