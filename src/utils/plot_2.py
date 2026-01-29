import numpy as np
import polars as pl
from matplotlib.collections import LineCollection
import settings as S

def add_svarasthana_guides_to_ax(
    ax,
    *,
    theoretical_cents: np.ndarray,
    found_cents: np.ndarray,
    threshold_cents: float,
    show_theoretical: bool = True,
    show_found: bool = True,
    show_band: bool = True,
):
    """
    Pitch vs time plot:
      - svarasthanes are HORIZONTAL lines (pitch axis)
    """

    if show_theoretical:
        for s in theoretical_cents:
            ax.axhline(
                float(s),
                color="gray",
                linestyle="--",
                alpha=0.8,
                linewidth=1.2,
                zorder=3,
            )

    if show_found:
        for s in found_cents:
            ax.axhline(
                float(s),
                color="red",
                linestyle="-",
                alpha=0.8,
                linewidth=0.9,
                zorder=2,
            )

    if show_band:
        for s in found_cents:
            ax.axhspan(
                float(s - threshold_cents),
                float(s + threshold_cents),
                color="red",
                alpha=0.12,
                zorder=0,
            )

    return ax



def add_flat_threshold_segments_to_ax(
    ax,
    df_pitch: pl.DataFrame,
    *,
    time_col: str = S.TIME_COL,
    pitch_cents_col: str = S.PITCH_COL_CENTS,
    flat_col: str = S.STABLE_COL,
    theoretical_cents: np.ndarray,
    threshold_cents: float = 30.0,
    base_lw: float = 1.5,
    thick_lw: float = 3.2,
):
    """
    Re-draw line segments (as LineCollection) so that segments where:
      flat_col==True AND exists a theoretical svara within threshold
    are drawn thicker.

    This is purely a helper overlay; it doesn't remove your existing plot.
    """
    if time_col not in df_pitch.columns or pitch_cents_col not in df_pitch.columns:
        raise ValueError("Missing time/pitch cents columns")

    dfp = (
        df_pitch
        .select([time_col, pitch_cents_col, flat_col])
        .sort(time_col)
        .filter(pl.col(time_col).is_finite())
    )

    x = dfp[time_col].to_numpy().astype(float)
    y = dfp[pitch_cents_col].to_numpy().astype(float)
    flat = dfp[flat_col].fill_null(False).to_numpy().astype(bool) if flat_col in dfp.columns else np.zeros(len(dfp), bool)

    # per-frame: "is within threshold of ANY theoretical svara"
    near = np.zeros_like(flat, dtype=bool)
    for i in range(len(y)):
        if not np.isfinite(y[i]):
            continue
        if not flat[i]:
            continue
        if np.any(np.abs(theoretical_cents - y[i]) <= float(threshold_cents)):
            near[i] = True

    segments = []
    widths = []
    colors = []

    for i in range(len(x) - 1):
        if not (np.isfinite(y[i]) and np.isfinite(y[i + 1])):
            continue

        is_flat = bool(flat[i] or flat[i + 1])
        is_near = bool(near[i] or near[i + 1])   # near ja implica flat, però ho deixem explícit

        segments.append([(x[i], y[i]), (x[i + 1], y[i + 1])])

        if is_flat and is_near:
            widths.append(thick_lw)
            colors.append("darkred")
        elif is_flat and (not is_near):
            widths.append(base_lw)
            colors.append("red")
        else:
            widths.append(base_lw)
            colors.append("gray")

    if segments:
        lc = LineCollection(
            segments,
            linewidths=widths,
            colors=colors,
            alpha=0.9,
            zorder=3,  
        )
        ax.add_collection(lc)

    return ax