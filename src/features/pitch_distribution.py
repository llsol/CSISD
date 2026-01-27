import numpy as np
import polars as pl
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks


def compute_pitch_kde_and_peaks(
    pitch_hz,
    bw_method=0.1,
    num_points=8000,
    peak_distance=50,
):
    """
    pitch_hz: pl.Series | array-like (Hz)

    Returns:
      pitch_grid_hz, density, peak_hz, peak_densities
    """
    x = np.asarray(pitch_hz.to_numpy() if isinstance(pitch_hz, pl.Series) else pitch_hz, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x > 0]

    kde = gaussian_kde(x, bw_method=bw_method)
    pitch_grid = np.linspace(x.min(), x.max(), num_points)
    density = kde(pitch_grid)

    peaks, _ = find_peaks(density, distance=peak_distance)
    peak_hz = pitch_grid[peaks]
    peak_densities = density[peaks]

    return pitch_grid, density, peak_hz, peak_densities


def peaks_df(peak_hz, peak_densities, recording_id=None):
    """
    Returns a Polars DataFrame with peaks for one recording.
    """
    df = pl.DataFrame(
        {
            "peak_hz": peak_hz,
            "density": peak_densities,
            "peak_id": [f"peak_{i}" for i in range(len(peak_hz))],
        }
    ).sort("peak_hz")

    if recording_id is not None:
        df = df.with_columns(pl.lit(recording_id).alias("recording_id"))

    return df


def compute_recording_peaks_df(
    df_pitch: pl.DataFrame,
    pitch_col: str = "pitch_hz",
    recording_id=None,
    bw_method=0.1,
    num_points=8000,
    peak_distance=50,
) -> pl.DataFrame:
    """
    Convenience wrapper: df_pitch (one recording) -> peaks DataFrame.
    """
    _, _, peak_hz, peak_dens = compute_pitch_kde_and_peaks(
        df_pitch[pitch_col],
        bw_method=bw_method,
        num_points=num_points,
        peak_distance=peak_distance,
    )
    return peaks_df(peak_hz, peak_dens, recording_id=recording_id)
