"""
Plot all GT svaras that contain a direct consecutive transition between two
configurable segment types.

Run examples:
    python -m src.analysis.plot_sta_cp_svaras                      # default: STA→CP
    python -m src.analysis.plot_sta_cp_svaras --from STAp,STAt --to SIL
    python -m src.analysis.plot_sta_cp_svaras --from CP --to SIL
    python -m src.analysis.plot_sta_cp_svaras --from TRa,TRd --to CP

Shorthands: "STA" expands to {"STAp", "STAt"}, "TR" expands to {"TRa", "TRd"}.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import settings as S
from src.io.pitch_io import load_preprocessed_pitch, load_flat_regions, load_peaks
from src.io.annotation_io import load_annotations
from src.features.structural_embedding import (
    label_samples_sil_cp_sta,
    map_peaks_to_global_rows,
    restrict_peaks_to_slice,
    build_segments_for_one_svara,
    assign_segment_cents,
)

OUT_DIR = Path("figures/structural_analysis/v1")

SEG_COLORS = {
    "CP":   "#4caf50",
    "SIL":  "#bdbdbd",
    "STAp": "#ff9800",
    "STAt": "#e65100",
    "TRa":  "#2196f3",
    "TRd":  "#0277bd",
}

_SHORTHANDS = {
    "STA": {"STAp", "STAt"},
    "TR":  {"TRa",  "TRd"},
}


def expand_types(spec: str) -> set[str]:
    """
    Parse a comma-separated type spec into a set of canonical segment type names.
    Shorthands: STA → {STAp, STAt}, TR → {TRa, TRd}.
    """
    out: set[str] = set()
    for tok in spec.split(","):
        tok = tok.strip()
        if tok in _SHORTHANDS:
            out |= _SHORTHANDS[tok]
        else:
            out.add(tok)
    return out


def load_matching_svaras(
    from_types: set[str],
    to_types: set[str],
) -> list[dict]:
    """
    For each recording, collect svaras that contain at least one consecutive
    (from_types[i] → to_types[i+1]) pair in their segment list.

    Returns list of dicts with keys:
        recording_id, svara_label, performer,
        time_sec, cents, segments, hit_pairs
    """
    matches = []

    for rid in S.RECORDING_SELECTION:
        print(f"  {rid}...")
        performer = rid.split("_")[2]
        tonic_hz  = S.RECORDING_SELECTION_TONICS[rid]

        df_pitch = load_preprocessed_pitch(rid, S.INTERIM_RECORDINGS, tonic_hz, convert_to_cents=True)
        df_flat  = load_flat_regions(recording_id=rid, root_dir=S.INTERIM_RECORDINGS)
        df_peaks = load_peaks(recording_id=rid, root_dir=S.INTERIM_RECORDINGS)

        df_pitch = (
            df_pitch
            .join(df_flat.select(["time_rel_sec", "flat_region"]), on="time_rel_sec", how="left")
            .with_columns(pl.col("flat_region").fill_null(False))
            .with_row_index("row_idx")
        )

        ann_path  = S.DATA_CORPUS / rid / "raw" / f"{rid}_ann_svara.tsv"
        df_svaras = load_annotations(file_path=ann_path, annotation_type="svara", engine="polars")
        peak_row_map = map_peaks_to_global_rows(df_pitch, df_peaks)
        t_all        = df_pitch["time_rel_sec"].to_numpy()

        for ann in df_svaras.iter_rows(named=True):
            t0 = float(ann["start_time_sec"])
            t1 = float(ann["end_time_sec"])

            mask     = (t_all >= t0) & (t_all <= t1)
            df_svara = df_pitch.filter(pl.Series(mask))
            if df_svara.is_empty():
                continue

            df_svara = label_samples_sil_cp_sta(df_svara)
            local_pm = restrict_peaks_to_slice(df_svara, peak_row_map)
            segs     = build_segments_for_one_svara(df_svara=df_svara, local_peak_map=local_pm)
            segs     = assign_segment_cents(segs, df_svara, local_pm)

            types = [s["type"] for s in segs]
            hit_pairs = [
                (i, i + 1)
                for i in range(len(types) - 1)
                if types[i] in from_types and types[i + 1] in to_types
            ]
            if not hit_pairs:
                continue

            matches.append({
                "recording_id": rid,
                "performer":    performer,
                "svara_label":  ann["svara_label"],
                "time_sec":     df_svara["time_rel_sec"].to_numpy(),
                "cents":        df_svara["f0_savgol_p3_w13_cents"].to_numpy(),
                "segments":     segs,
                "hit_pairs":    hit_pairs,
            })

    return matches


def plot_matches(
    matches: list[dict],
    from_types: set[str],
    to_types: set[str],
    out_path: Path,
) -> None:
    n = len(matches)
    ncols = 5
    nrows = max(1, (n + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 2.8))
    axes = np.array(axes).flatten()

    for ax, m in zip(axes, matches):
        t_all = m["time_sec"]
        t     = t_all - t_all[0]
        cents = m["cents"]
        segs  = m["segments"]

        # Background spans per segment
        for seg in segs:
            s_idx = seg["start"]
            e_idx = min(seg["end"], len(t_all) - 1)
            ax.axvspan(
                t_all[s_idx] - t_all[0],
                t_all[e_idx] - t_all[0],
                alpha=0.25,
                color=SEG_COLORS.get(seg["type"], "#eeeeee"),
                linewidth=0,
            )

        # Highlight matching transition spans
        for i, j in m["hit_pairs"]:
            seg_a = segs[i]
            seg_b = segs[j]
            t_a0 = t_all[seg_a["start"]] - t_all[0]
            t_ab = t_all[seg_b["start"]] - t_all[0]
            t_b1 = t_all[min(seg_b["end"], len(t_all) - 1)] - t_all[0]
            ax.axvspan(t_a0, t_ab, alpha=0.5,
                       color=SEG_COLORS.get(seg_a["type"], "#eeeeee"), linewidth=0, zorder=0)
            ax.axvspan(t_ab, t_b1, alpha=0.5,
                       color=SEG_COLORS.get(seg_b["type"], "#eeeeee"), linewidth=0, zorder=0)
            ax.axvline(t_ab, color="red", linewidth=1.2, linestyle="--", zorder=3)

        # Pitch curve
        voiced = cents != 0
        ax.plot(t[voiced], cents[voiced], color="black", linewidth=0.8, zorder=2)

        transition_str = " + ".join(
            f"{segs[i]['type']}→{segs[j]['type']}" for i, j in m["hit_pairs"]
        )
        ax.set_title(
            f"{m['svara_label']}  [{m['performer']}]\n{transition_str}",
            fontsize=7, pad=2,
        )
        ax.set_xlabel("t (s)", fontsize=6)
        ax.set_ylabel("cents", fontsize=6)
        ax.tick_params(labelsize=6)
        voiced_cents = cents[voiced & np.isfinite(cents)]
        if voiced_cents.size > 0:
            margin = max(50, (voiced_cents.max() - voiced_cents.min()) * 0.15)
            ax.set_ylim(voiced_cents.min() - margin, voiced_cents.max() + margin)

    for ax in axes[n:]:
        ax.set_visible(False)

    legend_patches = [
        mpatches.Patch(color=c, alpha=0.6, label=lbl)
        for lbl, c in SEG_COLORS.items()
    ]
    fig.legend(handles=legend_patches, loc="lower right",
               ncol=3, fontsize=7, framealpha=0.8)

    from_label = "|".join(sorted(from_types))
    to_label   = "|".join(sorted(to_types))
    fig.suptitle(
        f"GT svaras with {from_label}→{to_label} direct transition  (n={n})",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--from", dest="from_spec", default="TR",
        metavar="TYPES",
        help="Comma-separated source segment types (default: STA → STAp,STAt). "
             "Shorthands: STA, TR.",
    )
    parser.add_argument(
        "--to", dest="to_spec", default="CP",
        metavar="TYPES",
        help="Comma-separated target segment types (default: CP). "
             "Shorthands: STA, TR.",
    )
    args = parser.parse_args()

    from_types = expand_types(args.from_spec)
    to_types   = expand_types(args.to_spec)

    from_label = "_".join(sorted(from_types))
    to_label   = "_".join(sorted(to_types))

    print(f"Looking for {from_label} → {to_label} transitions...")
    matches = load_matching_svaras(from_types, to_types)
    print(f"\nFound {len(matches)} svaras")

    counts: dict[str, int] = {}
    for m in matches:
        for i, j in m["hit_pairs"]:
            key = f"{m['segments'][i]['type']}→{m['segments'][j]['type']}"
            counts[key] = counts.get(key, 0) + 1
    print("Breakdown:", counts)

    out_path = OUT_DIR / f"31_{from_label}__{to_label}_svaras.png"
    plot_matches(matches, from_types, to_types, out_path)


if __name__ == "__main__":
    main()
