import matplotlib.pyplot as plt
import numpy as np


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
