import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

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