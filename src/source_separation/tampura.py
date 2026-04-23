"""
Tampura drone attenuation via classical signal processing.

Pipeline:
  1. Load audio
  2. STFT
  3. Tampura mask — two modes:
       a) Reference segment (preferred): mean spectrum of the first N seconds,
          where only the tampura drone is present.
       b) Fallback: harmonic proximity × temporal stability × persistence.
  4. Soft attenuation
  5. ISTFT → attenuated audio

Steps not implemented here (run downstream):
  7. FTA-Net pitch extraction on the processed audio
  8. Pitch postprocessing
"""

from __future__ import annotations

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from scipy.ndimage import uniform_filter1d


# ---------------------------------------------------------------------------
# Mask builders
# ---------------------------------------------------------------------------

def _harmonic_mask(
    freqs: np.ndarray,
    tonic_hz: float,
    n_harmonics: int,
    bandwidth_cents: float,
) -> np.ndarray:
    """Soft Gaussian notches at tonic harmonics. Shape: (n_freqs,)."""
    mask = np.zeros(len(freqs))
    for h in range(1, n_harmonics + 1):
        f_h = tonic_hz * h
        bw_hz = f_h * (2 ** (bandwidth_cents / 1200) - 1)
        bell = np.exp(-0.5 * ((freqs - f_h) / (bw_hz / 2 + 1e-8)) ** 2)
        mask = np.maximum(mask, bell)
    return mask


def _stability_mask(S_mag: np.ndarray, smoothing_frames: int = 11) -> np.ndarray:
    """
    Inverse normalised temporal std per bin.
    Low variance over time → high score (stable = likely drone).
    Shape: (n_freqs,)
    """
    S_smooth = uniform_filter1d(S_mag, size=smoothing_frames, axis=1)
    std = S_smooth.std(axis=1)
    std_norm = std / (std.max() + 1e-8)
    return 1.0 - std_norm


def _persistence_mask(S_mag: np.ndarray, threshold_ratio: float = 0.15) -> np.ndarray:
    """
    Fraction of frames where a bin exceeds threshold_ratio × global max.
    Drone frequencies are active nearly all the time.
    Shape: (n_freqs,)
    """
    threshold = threshold_ratio * S_mag.max()
    return (S_mag > threshold).mean(axis=1)


# ---------------------------------------------------------------------------
# Tonic estimation (fallback when tonic_hz is not provided)
# ---------------------------------------------------------------------------

def estimate_tonic(
    S_mag: np.ndarray,
    freqs: np.ndarray,
    f_min: float = 60.0,
    f_max: float = 400.0,
) -> float:
    """
    Rough tonic estimate via harmonic product spectrum over the mean magnitude.
    Prefer passing the known tonic from settings.SARASUDA_TONICS.
    """
    mean_spec = S_mag.mean(axis=1)
    band = (freqs >= f_min) & (freqs <= f_max)
    freqs_b, spec_b = freqs[band], mean_spec[band]

    hps = spec_b.copy()
    for h in range(2, 5):
        hps *= np.interp(freqs_b * h, freqs, mean_spec, left=0.0, right=0.0)

    return float(freqs_b[np.argmax(hps)])


# ---------------------------------------------------------------------------
# Reference-segment mask
# ---------------------------------------------------------------------------

def _reference_mask(
    S_mag: np.ndarray,
    segments_frames: list[tuple[int, int]],
    log_scale: bool = True,
) -> np.ndarray:
    """
    Mean magnitude spectrum across all reference segments, normalised to [0,1].
    segments_frames : list of (start_frame, end_frame) — tampura-only regions.

    log_scale : if True, normalise in log-magnitude domain so high harmonics
                (which are weaker in amplitude) get a proportionally higher
                mask value and are actually attenuated.
    Shape: (n_freqs,)
    """
    chunks = [S_mag[:, start:end] for start, end in segments_frames if end > start]
    ref = np.concatenate(chunks, axis=1).mean(axis=1)
    if log_scale:
        ref = np.log1p(ref)
    mask = ref / (ref.max() + 1e-8)
    return mask


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def separate_tampura(
    audio_path: Path | str,
    output_path: Path | str,
    tonic_hz: float | None = None,
    *,
    reference_segments_sec: list[tuple[float, float]] | None = None,
    sr: int = 22050,
    n_fft: int = 4096,
    hop_length: int = 512,
    attenuation_db: float = 20.0,
    log_scale: bool = True,
    mask_gamma: float = 1.0,
    n_harmonics: int = 8,
    bandwidth_cents: float = 50.0,
    stability_weight: float = 0.5,
    persistence_threshold: float = 0.15,
) -> np.ndarray:
    """
    Attenuate the tampura drone from an audio recording.

    Parameters
    ----------
    audio_path              : input audio file
    output_path             : where to write the processed audio
    tonic_hz                : tonic in Hz; used only in fallback mode
    reference_segments_sec  : list of (start_sec, end_sec) intervals containing
                              only the tampura drone. Their mean spectrum becomes
                              the attenuation mask. Preferred over fallback.
    sr                      : sample rate
    n_fft                   : FFT size (4096 → ~5.4 Hz resolution at 22 kHz)
    hop_length              : STFT hop
    attenuation_db          : drone reduction in dB (20 dB = ×0.1 amplitude)
    n_harmonics             : harmonics to mask (fallback mode only)
    bandwidth_cents         : notch half-width in cents (fallback mode only)
    stability_weight        : stability vs persistence balance (fallback only)
    persistence_threshold   : active-frame threshold (fallback mode only)

    Returns
    -------
    y_clean : attenuated signal (also saved to output_path)
    """
    audio_path = Path(audio_path)
    output_path = Path(output_path)

    # 1. Load audio
    y, _ = librosa.load(audio_path, sr=sr, mono=True)

    # 2. STFT
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S_mag = np.abs(D)
    S_phase = np.angle(D)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # 3. Tampura mask
    if reference_segments_sec is not None:
        n_frames = S_mag.shape[1]
        segments_frames = [
            (
                min(int(t0 * sr / hop_length), n_frames),
                min(int(t1 * sr / hop_length), n_frames),
            )
            for t0, t1 in reference_segments_sec
        ]
        total_ref_frames = sum(e - s for s, e in segments_frames)
        drone_score = _reference_mask(S_mag, segments_frames, log_scale=log_scale)
        if mask_gamma != 1.0:
            drone_score = drone_score ** mask_gamma
        print(f"[tampura] reference mask: {len(segments_frames)} segment(s), {total_ref_frames} frames total, gamma={mask_gamma}")
    else:
        if tonic_hz is None:
            tonic_hz = estimate_tonic(S_mag, freqs)
            print(f"[tampura] estimated tonic: {tonic_hz:.1f} Hz")
        else:
            print(f"[tampura] tonic: {tonic_hz:.1f} Hz")

        h_mask = _harmonic_mask(freqs, tonic_hz, n_harmonics, bandwidth_cents)
        s_mask = _stability_mask(S_mag)
        p_mask = _persistence_mask(S_mag, persistence_threshold)
        drone_score = h_mask * (
            stability_weight * s_mask + (1.0 - stability_weight) * p_mask
        )
        drone_score /= drone_score.max() + 1e-8
        print("[tampura] fallback mask (harmonic + stability + persistence)")

    # 4. Soft attenuation
    alpha = 1.0 - 10 ** (-attenuation_db / 20.0)
    gain_1d = 1.0 - alpha * drone_score          # (n_freqs,)
    S_mag_clean = S_mag * gain_1d[:, np.newaxis]  # broadcast over frames

    # 5. ISTFT
    D_clean = S_mag_clean * np.exp(1j * S_phase)
    y_clean = librosa.istft(D_clean, hop_length=hop_length, length=len(y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, y_clean, sr)
    print(f"[tampura] written → {output_path}")

    return y_clean
