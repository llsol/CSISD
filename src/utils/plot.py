# Usage:
# fig, ax = plot_svara_window_with_flat_regions(...); add_peaks_to_ax(ax, df_peaks); plt.show()
# fig, ax = plot_pitchcurve_svara_window(...); add_peaks_to_ax(ax, df_peaks); plt.show()
# fig, axes = plot_segment_multiplot(...); add_peaks_to_ax(axes[0], df_peaks); plt.show()
# fig, ax = plot_pitch_kde_with_svaras(...); plt.show()
# fig, ax = plot_pitch_with_flat_regions(...); add_peaks_to_ax(ax, df_peaks); plt.show()
# add_peaks_to_ax(ax, df_peaks)


import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.collections import LineCollection
import settings as S


def plot_svara_window_with_flat_regions(
    df_pitch: pl.DataFrame,
    svara_id: int,
    *,
    time_col: str = "time_rel_sec",
    pitch_col: str = "f0_savgol_p3_w13",
    stable_col: str = "flat_region",
    svara_id_col: str = "svara_id",
    svara_start_label_col: str = "svara_start_label",
    svara_end_label_col: str = "svara_end_label",
    window_n: int = 1,
    figsize: tuple = (12, 4),
):
    # --- 1) calcula window ids com ja fas ---
    if window_n < 0:
        raise ValueError("window_n ha de ser >= 0")

    left = int((window_n + 1) // 2)
    right = int(window_n // 2)

    ids = (
        df_pitch
        .select(pl.col(svara_id_col).drop_nulls().unique().sort())
        .get_column(svara_id_col)
        .to_list()
    )
    if len(ids) == 0:
        raise ValueError("No hi ha cap svara_id al df_pitch")
    if svara_id not in ids:
        raise ValueError(f"svara_id={svara_id} no existeix. Rang: {ids[0]}..{ids[-1]}")

    k = ids.index(svara_id)
    k0 = max(0, k - left)
    k1 = min(len(ids) - 1, k + right)
    window_ids = ids[k0:k1 + 1]

    df_win = (
    df_pitch
    .filter(pl.col(svara_id_col).is_in(window_ids))
    .sort(time_col)
    .filter(pl.col(time_col).is_finite())
)

    if df_win.height < 2:
        return None, None

    # --- 2) crea fig/ax ---
    fig, ax = plt.subplots(figsize=figsize)

    # --- 3) pinta flat regions al mateix ax ---
    x, y = _add_flat_regions_to_ax(
        ax, df_win,
        time_col=time_col,
        pitch_col=pitch_col,
        stable_col=stable_col,
    )

    # --- 4) boundaries start/end (igual que abans) ---
    if svara_start_label_col in df_win.columns:
        df_starts = df_win.filter(pl.col(svara_start_label_col).is_not_null())
        if not df_starts.is_empty():
            for ts in df_starts.select(pl.col(time_col).unique().sort()).get_column(time_col).to_list():
                ax.axvline(float(ts), linestyle="--")

    if svara_end_label_col in df_win.columns:
        df_ends = df_win.filter(pl.col(svara_end_label_col).is_not_null())
        if not df_ends.is_empty():
            for ts in df_ends.select(pl.col(time_col).unique().sort()).get_column(time_col).to_list():
                ax.axvline(float(ts), linestyle="--")

    # --- 5) labels ---
    ax.set_title(f"{pitch_col} — svara_id={svara_id} (window: {window_ids[0]}..{window_ids[-1]})")
    ax.set_xlabel(time_col)
    ax.set_ylabel(pitch_col)

    # límits
    x_finite = x[np.isfinite(x)]
    if x_finite.size >= 2:
        ax.set_xlim(float(x_finite[0]), float(x_finite[-1]))

    if np.any(np.isfinite(y)):
        ax.set_ylim(float(np.nanmin(y)) - 50, float(np.nanmax(y)) + 50)

    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    # plt.show()
    return fig, ax


# ----------------------------
# 1) Pitchcurve + svara window
# ----------------------------
def plot_pitchcurve_svara_window(
    df_pitch: pl.DataFrame,
    svara_id: int,
    *,
    time_col: str = "time_rel_sec",
    pitch_col: str = "f0_savgol_p3_w13",
    svara_id_col: str = "svara_id",
    svara_start_label_col: str = "svara_start_label",
    svara_end_label_col: str = "svara_end_label",
    window_n: int = 1,
    figsize: tuple = (12, 4),
):
    """
    Ploteja una finestra de svaras al voltant d'un `svara_id`.

    window_n:
      0 -> només la svara
      1 -> svara + anterior
      2 -> anterior + posterior
      3 -> 2 anteriors + posterior
      4 -> 2 anteriors + 2 posteriors
    (generalització: left=ceil(window_n/2), right=floor(window_n/2))

    Marca amb línies verticals discontínues:
      - tots els starts (svara_start_label != null)
      - tots els ends   (svara_end_label != null)

    Retorna (fig, ax)
    """
    if time_col not in df_pitch.columns:
        raise ValueError(f"df_pitch no té columna '{time_col}'")
    if pitch_col not in df_pitch.columns:
        raise ValueError(f"df_pitch no té columna '{pitch_col}'")
    if svara_id_col not in df_pitch.columns:
        raise ValueError(f"df_pitch no té columna '{svara_id_col}'")

    if window_n < 0:
        raise ValueError("window_n ha de ser >= 0")

    left = int((window_n + 1) // 2)   # ceil(window_n/2)
    right = int(window_n // 2)        # floor(window_n/2)

    ids = (
        df_pitch
        .select(pl.col(svara_id_col).drop_nulls().unique().sort())
        .get_column(svara_id_col)
        .to_list()
    )
    if len(ids) == 0:
        raise ValueError("No hi ha cap svara_id al df_pitch (tots són null?)")
    if svara_id not in ids:
        raise ValueError(f"svara_id={svara_id} no existeix. Rang: {ids[0]}..{ids[-1]}")

    k = ids.index(svara_id)
    k0 = max(0, k - left)
    k1 = min(len(ids) - 1, k + right)
    window_ids = ids[k0:k1 + 1]

    df_win = df_pitch.filter(pl.col(svara_id_col).is_in(window_ids)).sort(time_col)
    if df_win.is_empty():
        raise ValueError("Finestra buida (inesperat)")

    t = df_win.get_column(time_col).to_numpy()
    y = df_win.get_column(pitch_col).to_numpy()

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(t, y)

    ax.set_title(f"{pitch_col} — svara_id={svara_id} (window: {window_ids[0]}..{window_ids[-1]})")
    ax.set_xlabel(time_col)
    ax.set_ylabel(pitch_col)

    # --- marca starts ---
    if svara_start_label_col in df_win.columns:
        df_starts = df_win.filter(pl.col(svara_start_label_col).is_not_null())
        if not df_starts.is_empty():
            start_times = (
                df_starts
                .select(pl.col(time_col).unique().sort())
                .get_column(time_col)
                .to_list()
            )
            for ts in start_times:
                ax.axvline(float(ts), linestyle="--")

    # --- marca ends ---
    if svara_end_label_col in df_win.columns:
        df_ends = df_win.filter(pl.col(svara_end_label_col).is_not_null())
        if not df_ends.is_empty():
            end_times = (
                df_ends
                .select(pl.col(time_col).unique().sort())
                .get_column(time_col)
                .to_list()
            )
            for ts in end_times:
                ax.axvline(float(ts), linestyle="--")

    ax.margins(x=0.01)
    fig.tight_layout()
    return fig, ax


# ----------------------------
# 2) Multiplot for segments
# ----------------------------
def plot_segment_multiplot(
    df,
    *,
    time_col: str = "time_rel_sec",
    main_cols="f0_savgol_p3_w13",
    sub_cols=None,
    bool_markers=None,
    superpose=None,
    figsize=(12, 8),
    sharex: bool = True,
    colors=None,
    styles=None,
    fill_under=None,
    title=None,
    time_range=None,   # (t0, t1)
):
    """
    Plot a pitch segment with multiple subplots.

    time_range : (t0, t1) or None
        If not None, restrict plotting to t in [t0, t1].
    """

    # Keep original behavior: accept pandas or polars
    if hasattr(df, "to_pandas"):
        pdf = df.to_pandas()
    else:
        pdf = df.copy()

    # TIME RANGE FILTER
    if time_range is not None:
        t0, t1 = time_range
        mask = (pdf[time_col] >= t0) & (pdf[time_col] <= t1)
        pdf = pdf.loc[mask]

    t = pdf[time_col].to_numpy()

    # Normalize main_cols to list (original code assumes iterable)
    if isinstance(main_cols, str):
        main_cols = [main_cols]
    else:
        main_cols = list(main_cols)

    # Normalize sub_cols
    if sub_cols is None:
        sub_cols = []
    if isinstance(sub_cols, list) and len(sub_cols) > 0 and isinstance(sub_cols[0], str):
        sub_cols = [[v] for v in sub_cols]

    n_subplots = 1 + len(sub_cols)
    fig, axes = plt.subplots(n_subplots, 1, figsize=figsize, sharex=sharex)
    if n_subplots == 1:
        axes = [axes]

    def _plot_vars(ax, var_list):
        for var in var_list:
            y = pdf[var].to_numpy()
            c = colors.get(var) if colors else None
            ls = styles.get(var, "-") if styles else "-"
            ax.plot(t, y, label=var, color=c, linestyle=ls)

            if fill_under and var in fill_under:
                ax.fill_between(t, y, alpha=0.3, color=fill_under[var])

        ax.legend(loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.4)

    # MAIN subplot
    ax0 = axes[0]
    _plot_vars(ax0, main_cols)

    if superpose:
        for mv, extras in superpose.items():
            if mv in main_cols:
                _plot_vars(ax0, extras)

    if bool_markers:
        for bool_col, style in bool_markers.items():
            if bool_col not in pdf.columns:
                continue
            mask = pdf[bool_col].to_numpy().astype(bool)
            x = t[mask]
            for var in main_cols:
                y = pdf[var].to_numpy()[mask]
                ax0.scatter(
                    x, y,
                    color=style.get("color", "red"),
                    marker=style.get("marker", "o"),
                    s=style.get("size", 20),
                )

    # SUBPLOTS
    for i, vars_i in enumerate(sub_cols):
        ax = axes[i + 1]
        _plot_vars(ax, vars_i)

        if superpose:
            for mv, extras in superpose.items():
                if mv in vars_i:
                    _plot_vars(ax, extras)

        if bool_markers:
            for bool_col, style in bool_markers.items():
                if bool_col not in pdf.columns:
                    continue
                mask = pdf[bool_col].to_numpy().astype(bool)
                x = t[mask]
                for var in vars_i:
                    y = pdf[var].to_numpy()[mask]
                    ax.scatter(
                        x, y,
                        color=style.get("color", "red"),
                        marker=style.get("marker", "o"),
                        s=style.get("size", 20),
                    )

        ax.grid(True, linestyle="--", alpha=0.4)

    if title:
        fig.suptitle(title)

    plt.xlabel(time_col)
    plt.tight_layout()
    # plt.show()
    return fig, axes


# 4) KDE with svara grid

def plot_pitch_kde_with_svaras(
    pitch_grid_cents: np.ndarray,
    density: np.ndarray,
    peak_cents: np.ndarray,
    peak_densities: np.ndarray,
    *,
    peaks_df: pl.DataFrame | None = None,
    svara_labels = S.RAGAM_SVARAS_CENTS["saveri"]["svarasthanas"],
    intervals_cents = S.RAGAM_SVARAS_CENTS["saveri"]["intervals_cents"],
):
    """
    KDE in cents + detected peaks + theoretical svara grid.
    """
    base_cents = np.cumsum(intervals_cents)

    cents_positions = []
    svaras_full = []
    for shift in (-1200, 0, 1200):
        for label, pos in zip(svara_labels, base_cents):
            cents_positions.append(pos + shift)
            if shift == -1200:
                svaras_full.append(label + ",")
            elif shift == 0:
                svaras_full.append(label)
            else:
                svaras_full.append(label + "'")

    plt.figure(figsize=(14, 4))
    plt.plot(pitch_grid_cents, density, label="Pitch KDE")
    plt.plot(peak_cents, peak_densities, "ro", label="Detected peaks")

    plt.title("Peak pitch density detection (svarasthana candidates)")
    plt.xlabel("Cents (relative to tonic)")
    plt.ylabel("Density")
    plt.grid(True, linestyle="--", alpha=0.5)

    # theoretical grid
    for pos in cents_positions:
        plt.axvline(x=pos, color="gray", linestyle="--", alpha=0.4)

    plt.xticks(cents_positions, svaras_full, fontsize=9)

    # optional annotation (from DF)
    if peaks_df is not None and "peak_cents" in peaks_df.columns:
        for c in peaks_df["peak_cents"].to_numpy():
            idx = np.abs(pitch_grid_cents - c).argmin()
            plt.text(
                float(c),
                float(density[idx]),
                f"{c:.0f}",
                fontsize=8,
                color="gray",
                ha="left",
                va="bottom",
            )

    plt.legend()
    plt.tight_layout()
    # plt.show()
    return plt.gcf(), plt.gca()


def _add_flat_regions_to_ax(
    ax,
    dfp: pl.DataFrame,
    *,
    time_col: str,
    pitch_col: str,
    stable_col: str,
):
    x = dfp[time_col].to_numpy()
    y = dfp[pitch_col].to_numpy()

    stable = (
        dfp.select(pl.col(stable_col).fill_null(False).cast(pl.Boolean))
           .to_series()
           .to_numpy()
    ) if stable_col in dfp.columns else np.zeros(len(dfp), dtype=bool)

    segments, colors = [], []
    for i in range(len(x) - 1):
        if np.isnan(y[i]) or np.isnan(y[i + 1]):
            continue
        segments.append([(x[i], y[i]), (x[i + 1], y[i + 1])])
        is_stable_seg = bool(stable[i]) or bool(stable[i + 1])
        colors.append("crimson" if is_stable_seg else "lightgrey")

    if segments:
        lc = LineCollection(segments, colors=colors, linewidths=1.8)
        ax.add_collection(lc)

    return x, y

def plot_pitch_with_flat_regions(
    df: pl.DataFrame,
    *,
    time_col: str = "time_rel_sec",
    pitch_col: str = "f0_savgol_p3_w13",
    stable_col: str = "flat_region",
    title: str | None = None,
    time_range: tuple[float, float] | None = None,
    show_threshold_lines: bool = True,
    pitch_unit: str = "cents",
):
    if time_range is not None:
        t_start, t_end = time_range
        dfp = df.filter((pl.col(time_col) >= t_start) & (pl.col(time_col) <= t_end))
    else:
        dfp = df
        t_start = dfp.select(pl.col(time_col).min()).item()
        t_end = dfp.select(pl.col(time_col).max()).item()

    if dfp.height < 2:
        return None, None

    fig_width, dpi = (16, 100)
    if (t_end - t_start) < 1.0:
        fig_width, dpi = (14, 200)

    fig, ax = plt.subplots(figsize=(fig_width, 5), dpi=dpi)

    x, y = _add_flat_regions_to_ax(
        ax, dfp,
        time_col=time_col,
        pitch_col=pitch_col,
        stable_col=stable_col,
    )

    ax.set_xlim(t_start, t_end)
    if np.any(np.isfinite(y)):
        y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
        pad = 50 if pitch_unit == "cents" else 0.0
        ax.set_ylim(y_min - pad, y_max + pad)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"Pitch ({pitch_unit})")
    if title:
        ax.set_title(title)

    if show_threshold_lines:
        ax.axvline(t_start, linestyle=":", color="gray", linewidth=0.8)
        ax.axvline(t_end, linestyle=":", color="gray", linewidth=0.8)

    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    #plt.show()
    return fig, ax




def add_peaks_to_ax(
    ax,
    df_peaks: pl.DataFrame,
    *,
    show: str = "both",      # "savgol" | "raw" | "both"
    kind: str = "both",      # "max" | "min" | "both"
):
    """
    Superposa els extrems sobre un ax existent (pitch curve en cents).
    """
    d = df_peaks
    if kind != "both" and "extremum_kind" in d.columns:
        d = d.filter(pl.col("extremum_kind") == kind)

    if d.is_empty():
        return ax

    if show in ("savgol", "both"):
        ax.scatter(
            d["time_savgol"].to_numpy(),
            d["value_savgol_cents"].to_numpy(),
            marker="o",
            label="extrema savgol",
        )

    if show in ("raw", "both"):
        ax.scatter(
            d["time_raw"].to_numpy(),
            d["value_raw_cents"].to_numpy(),
            marker="x",
            label="extrema raw",
        )

    return ax