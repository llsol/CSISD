import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.collections import LineCollection

def plot_segment_multiplot(
    df,
    main_cols,
    sub_cols=None,
    time_col="time_rel_sec",
    bool_markers=None,
    bool_marker_style=None,
    superpose=None,
    figsize=(12, 8),
    sharex=True,
    colors=None,
    styles=None,
    fill_under=None,
    title=None,
    time_window=None,   # <-- (t0, duracio)
):
    """
    Plot a pitch segment with multiple subplots.

    New:
    ----
    time_window : (t0, duracio) or None
        If not None, restrict plotting to:
            t in [t0, t0 + duracio]
    """

    # Convert to pandas if needed
    if hasattr(df, "to_pandas"):
        pdf = df.to_pandas()
    else:
        pdf = df.copy()

    # ----------------------------
    # TIME WINDOW FILTER (NEW)
    # ----------------------------
    if time_window is not None:
        t0, dur = time_window
        t1 = t0 + dur
        mask = (pdf[time_col] >= t0) & (pdf[time_col] <= t1)
        pdf = pdf.loc[mask]

    t = pdf[time_col].to_numpy()

    # Normalize sub_cols
    if sub_cols is None:
        sub_cols = []

    if isinstance(sub_cols, list) and len(sub_cols) > 0 and isinstance(sub_cols[0], str):
        sub_cols = [[v] for v in sub_cols]

    n_subplots = 1 + len(sub_cols)
    fig, axes = plt.subplots(n_subplots, 1, figsize=figsize, sharex=sharex)

    if n_subplots == 1:
        axes = [axes]

    # Helper
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

    # ----------------------------
    # MAIN subplot
    # ----------------------------
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

    # ----------------------------
    # SUBPLOTS
    # ----------------------------
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
    plt.show()




def plot_pitch_with_flat_regions(
    df: pl.DataFrame,
    time_col: str = "time_rel_sec",
    pitch_col: str = "f0_savgol_p3_w13",
    stable_col: str = "flat_region",
    title: str | None = None,
    zoom_between: tuple[float, float] | None = None,
    show_threshold_lines: bool = True,
):
    # --- 1) Zoom (Polars-only) ---
    if zoom_between is not None:
        t_start, t_end = zoom_between
        df = df.filter((pl.col(time_col) >= t_start) & (pl.col(time_col) <= t_end))
    else:
        # min/max en polars (pot retornar None si no hi ha files)
        t_start = df.select(pl.col(time_col).min()).item()
        t_end = df.select(pl.col(time_col).max()).item()

    # Si no hi ha dades després del zoom
    if df.height < 2:
        return None, None

    # --- extreure arrays (sense pandas) ---
    x = df[time_col].to_numpy()
    y = df[pitch_col].to_numpy()

    # stable booleà, amb null -> False
    stable = (
        df.select(pl.col(stable_col).fill_null(False).cast(pl.Boolean))
          .to_series()
          .to_numpy()
    )

    # --- 3) blindatge: si tot y és NaN en el zoom ---
    if np.all(np.isnan(y)):
        return None, None

    # --- segments per LineCollection ---
    segments = []
    colors = []
    for i in range(len(x) - 1):
        if np.isnan(y[i]) or np.isnan(y[i + 1]):
            continue

        seg = [(x[i], y[i]), (x[i + 1], y[i + 1])]
        segments.append(seg)

        # --- 2) estable si stable[i] & stable[i+1] ---
        is_stable_seg = bool(stable[i]) and bool(stable[i + 1])
        colors.append("crimson" if is_stable_seg else "lightgrey")

    # Si no hi ha segments (p.ex. tot NaN excepte un punt)
    if not segments:
        return None, None

    lc = LineCollection(segments, colors=colors, linewidths=1.8)

    fig_width = 16
    dpi = 100
    if (t_end - t_start) < 1.0:
        fig_width = 14
        dpi = 200

    fig, ax = plt.subplots(figsize=(fig_width, 5), dpi=dpi)
    ax.add_collection(lc)
    ax.set_xlim(t_start, t_end)

    y_min = float(np.nanmin(y))
    y_max = float(np.nanmax(y))
    ax.set_ylim(y_min - 50, y_max + 50)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pitch (cents)")
    if title:
        ax.set_title(title)

    if show_threshold_lines:
        # center_val = np.nanmedian(y)  # (no usat ara)
        ax.axvline(t_start, linestyle=":", color="gray", linewidth=0.8)
        ax.axvline(t_end, linestyle=":", color="gray", linewidth=0.8)

    ax.grid(True, linestyle="--", alpha=0.5)

    # svaras at Y-axis (igual que abans)
    svara_labels = ["S", "R2", "G2", "M1", "P", "D1", "D2", "N2", "S"]
    intervals = [0, 200, 100, 200, 200, 100, 100, 100, 200]
    base_cents = np.cumsum(intervals)

    svara_y = []
    svara_labels_full = []
    for shift, mark in [(-1200, ","), (0, ""), (1200, "'")]:
        for svara, pos in zip(svara_labels, base_cents):
            y_pos = pos + shift
            if ax.get_ylim()[0] <= y_pos <= ax.get_ylim()[1]:
                svara_y.append(y_pos)
                svara_labels_full.append(svara + mark)

    ax.set_yticks(svara_y)
    ax.set_yticklabels(svara_labels_full)

    plt.tight_layout()
    plt.show()
    return fig, ax
