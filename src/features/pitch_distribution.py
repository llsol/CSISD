import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
import settings as S

def hz_to_cents(f_hz: np.ndarray, tonic_hz: float) -> np.ndarray:
    f_hz = np.asarray(f_hz, dtype=float)
    out = np.full_like(f_hz, np.nan, dtype=float)
    mask = np.isfinite(f_hz) & (f_hz > 0)
    out[mask] = 1200.0 * np.log2(f_hz[mask] / float(tonic_hz))
    return out

def cents_to_hz(cents: np.ndarray, tonic_hz: float) -> np.ndarray:
    cents = np.asarray(cents, dtype=float)
    out = np.full_like(cents, np.nan, dtype=float)
    mask = np.isfinite(cents)
    out[mask] = float(tonic_hz) * (2.0 ** (cents[mask] / 1200.0))
    return out


def compute_pitch_kde_and_peaks(
    pitch,
    *,
    pitch_unit: str = "hz",         # "hz" o "cents"
    tonic_hz: float | None = None,  # necessari si pitch_unit="hz" i vols cents; o si pitch_unit="cents" i vols hz
    bw_method=0.1,
    num_points=8000,
    peak_distance=50,
):
    """
    pitch: pl.Series | array-like (Hz o cents segons pitch_unit)

    Retorna sempre:
      grid_cents, density, peak_cents, peak_densities

    I opcionalment (si tonic_hz està disponible):
      grid_hz, peak_hz
    """

    if pitch_unit not in ("hz", "cents"):
        raise ValueError("pitch_unit must be 'hz' or 'cents'")

    x = np.asarray(pitch.to_numpy() if isinstance(pitch, pl.Series) else pitch, dtype=float)
    x = x[np.isfinite(x)]

    if pitch_unit == "hz":
        x = x[x > 0]

    if x.size < 5:
        empty = np.array([], dtype=float)
        return empty, empty, empty, empty, empty, empty

    # KDE sobre la unitat d'entrada
    kde = gaussian_kde(x, bw_method=bw_method)
    grid = np.linspace(x.min(), x.max(), int(num_points))
    density = kde(grid)

    peaks_idx, _ = find_peaks(density, distance=int(peak_distance))
    peak_vals = grid[peaks_idx]
    peak_densities = density[peaks_idx]

    # Converteix a cents (sempre)
    if pitch_unit == "cents":
        grid_cents = grid
        peak_cents = peak_vals
        if tonic_hz is not None:
            grid_hz = cents_to_hz(grid_cents, tonic_hz=tonic_hz)
            peak_hz = cents_to_hz(peak_cents, tonic_hz=tonic_hz)
        else:
            grid_hz = np.array([], dtype=float)
            peak_hz = np.array([], dtype=float)

    else:
        # pitch_unit == "hz"
        if tonic_hz is None:
            raise ValueError("tonic_hz is required when pitch_unit='hz' (to get cents output)")
        grid_hz = grid
        peak_hz = peak_vals
        grid_cents = hz_to_cents(grid_hz, tonic_hz=tonic_hz)
        peak_cents = hz_to_cents(peak_hz, tonic_hz=tonic_hz)

    return grid_cents, density, peak_cents, peak_densities, grid_hz, peak_hz


def peaks_df(
    peak_cents: np.ndarray,
    peak_densities: np.ndarray,
    *,
    peak_hz: np.ndarray | None = None,
    recording_id: str | None = None,
) -> pl.DataFrame:

    data = {
        "peak_cents": peak_cents,
        "density": peak_densities,
        "peak_id": [f"peak_{i}" for i in range(len(peak_cents))],
    }
    if peak_hz is not None and len(peak_hz) == len(peak_cents):
        data["peak_hz"] = peak_hz

    df = pl.DataFrame(data).sort("peak_cents")

    if recording_id is not None:
        df = df.with_columns(pl.lit(recording_id).alias("recording_id"))

    return df


def compute_pitch_distribution_peaks_df(
    df_pitch: pl.DataFrame,
    *,
    pitch_col: str,
    pitch_unit: str = "hz",          # "hz" o "cents"
    tonic_hz: float | None = None,   # obligatori si pitch_unit="hz"
    recording_id: str | None = None,
    bw_method=0.1,
    num_points=8000,
    peak_distance=50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pl.DataFrame]:
    """
    Retorna:
      grid_cents, density, peak_cents, peak_densities, df_peaks
    """

    grid_cents, density, peak_cents, peak_dens, grid_hz, peak_hz = compute_pitch_kde_and_peaks(
        df_pitch[pitch_col],
        pitch_unit=pitch_unit,
        tonic_hz=tonic_hz,
        bw_method=bw_method,
        num_points=num_points,
        peak_distance=peak_distance,
    )

    df_peaks = peaks_df(
        peak_cents=peak_cents,
        peak_densities=peak_dens,
        peak_hz=(peak_hz if (peak_hz.size > 0) else None),
        recording_id=recording_id,
    )

    return grid_cents, density, peak_cents, peak_dens, df_peaks





def theoretical_svarasthanas_cents(
    *,
    raga: str = "saveri",
    with_octaves: bool = True,
) -> np.ndarray:
    base = np.cumsum(S.RAGAM_SVARAS_CENTS[raga]["intervals_cents"]).astype(float)  # 0..1200
    if not with_octaves:
        return base
    return np.sort(np.concatenate([base - 1200.0, base, base + 1200.0]))


def found_svarasthanas_from_kde(
    df_kde_peaks: pl.DataFrame,
    *,
    theoretical_cents: np.ndarray,
    n_largest: int = 5,
    threshold_cents: float = 30.0,
) -> pl.DataFrame:
    """
    Map KDE peaks (peak_cents, density) to nearby theoretical svarasthanas within threshold.
    Keeps top n_largest peaks by density.
    Returns: columns [source, peak_cents, density, svara_cents, dist_cents]
    """
    if df_kde_peaks.is_empty():
        print("Warning: df_kde_peaks is empty")
        return pl.DataFrame([])

    if "peak_cents" not in df_kde_peaks.columns or "density" not in df_kde_peaks.columns:
        raise ValueError("df_kde_peaks must have columns: peak_cents, density")

    top = df_kde_peaks.sort("density", descending=True).head(int(n_largest))
    peaks = top["peak_cents"].to_numpy().astype(float)
    dens = top["density"].to_numpy().astype(float)

    rows = []
    for p, d in zip(peaks, dens):
        dist = np.abs(theoretical_cents - p)
        mask = dist <= float(threshold_cents)
        for s, dd in zip(theoretical_cents[mask], dist[mask]):
            rows.append({
                "source": "kde_peaks",
                "peak_cents": float(p),
                "density": float(d),
                "svara_cents": float(s),
                "dist_cents": float(dd),
            })

    if not rows:
        return pl.DataFrame([])

    return pl.DataFrame(rows).sort(["svara_cents", "dist_cents"])


def flat_region_svara_proportions(
    df_pitch: pl.DataFrame,
    *,
    time_col: str = S.TIME_COL,
    pitch_cents_col: str = S.PITCH_COL_CENTS,
    flat_col: str = S.STABLE_COL,          # "flat_region"
    flat_id_col: str = "flat_id",
    theoretical_cents: np.ndarray,
    threshold_cents: float = 30.0,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Frame-level:
      - keep only frames where flat_col==True
      - for each frame, mark all theoretical svaras within threshold
    Returns:
      df_hits: one row per (frame, svara) hit [time, pitch, flat_id, svara_cents, dist_cents]
      df_props: proportions per flat_id and svara [flat_id, svara_cents, n_frames, duration_sec, proportion]
    """
    if flat_col not in df_pitch.columns:
        raise ValueError(f"Missing {flat_col} in df_pitch")
    if pitch_cents_col not in df_pitch.columns:
        raise ValueError(f"Missing {pitch_cents_col} in df_pitch")
    if time_col not in df_pitch.columns:
        raise ValueError(f"Missing {time_col} in df_pitch")
    if flat_id_col not in df_pitch.columns:
        raise ValueError(f"Missing {flat_id_col} (run add_flat_id first)")

    df_flat = (
        df_pitch
        .filter(pl.col(flat_col).fill_null(False) == True)
        .select([time_col, pitch_cents_col, flat_id_col])
        .drop_nulls([time_col, pitch_cents_col, flat_id_col])
        .sort(time_col)
    )
    if df_flat.is_empty():
        return pl.DataFrame([]), pl.DataFrame([])

    t = df_flat[time_col].to_numpy().astype(float)
    y = df_flat[pitch_cents_col].to_numpy().astype(float)
    fid = df_flat[flat_id_col].to_numpy()

    rows = []
    for ti, yi, fidi in zip(t, y, fid):
        dist = np.abs(theoretical_cents - yi)
        mask = dist <= float(threshold_cents)
        for s, dd in zip(theoretical_cents[mask], dist[mask]):
            rows.append({
                time_col: float(ti),
                pitch_cents_col: float(yi),
                flat_id_col: int(fidi),
                "source": "flat_regions",
                "svara_cents": float(s),
                "dist_cents": float(dd),
            })

    if not rows:
        return pl.DataFrame([]), pl.DataFrame([])

    df_hits = pl.DataFrame(rows)

    # duration estimate from time step (robust enough for your 100 Hz streams)
    dt = np.nanmedian(np.diff(np.unique(t))) if len(np.unique(t)) > 2 else np.nan
    dt = float(dt) if np.isfinite(dt) and dt > 0 else 0.0

    df_counts = (
        df_hits
        .group_by([flat_id_col, "svara_cents"])
        .agg(pl.len().alias("n_frames"))
        .with_columns((pl.col("n_frames") * dt).alias("duration_sec"))
    )

    df_tot = (
        df_counts
        .group_by(flat_id_col)
        .agg(pl.col("n_frames").sum().alias("n_frames_total"))
    )

    df_props = (
        df_counts
        .join(df_tot, on=flat_id_col, how="left")
        .with_columns((pl.col("n_frames") / pl.col("n_frames_total")).alias("proportion"))
        .sort([flat_id_col, "proportion"], descending=[False, True])
    )

    return df_hits, df_props