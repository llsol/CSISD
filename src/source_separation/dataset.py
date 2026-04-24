"""
Dataset per entrenament del U-Net de separació de tampura.

Estructura de dades esperada:
    data/source_separation/
        voice/      ← gravacions de veu neta   (.wav)
        tampura/    ← gravacions de tampura neta (.wav)

Cada __getitem__ retorna un patch aleatori de la mescla sintètica:
    mix = voice + alpha * tampura
amb alpha mostrejat aleatòriament dins [alpha_min, alpha_max].
"""

import random
from pathlib import Path

import numpy as np
import torch
import librosa
from torch.utils.data import Dataset


class TampuraSeparationDataset(Dataset):
    """
    Args:
        voice_dir    : carpeta amb .wav de veu neta
        tampura_dir  : carpeta amb .wav de tampura neta
        sr           : sample rate (tots els àudios es resamplen)
        n_fft        : mida de la FFT
        hop_length   : hop de la STFT
        patch_frames : nombre de frames per patch (dimensió temporal del tensor)
        alpha_range  : (min, max) del factor d'escala de la tampura
        augment      : si True, aplica petit pitch shift i time stretch aleatoris
    """

    def __init__(
        self,
        voice_dir: Path | str,
        tampura_dir: Path | str,
        sr: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        patch_frames: int = 128,
        alpha_range: tuple[float, float] = (0.3, 1.0),
        augment: bool = True,
    ):
        self.voice_files   = sorted(Path(voice_dir).glob("*.wav"))
        self.tampura_files = sorted(Path(tampura_dir).glob("*.wav"))

        if not self.voice_files:
            raise FileNotFoundError(f"No s'han trobat .wav a {voice_dir}")
        if not self.tampura_files:
            raise FileNotFoundError(f"No s'han trobat .wav a {tampura_dir}")

        self.sr           = sr
        self.n_fft        = n_fft
        self.hop_length   = hop_length
        self.patch_frames = patch_frames
        self.alpha_range  = alpha_range
        self.augment      = augment

        # Càrrega lazy: els àudios es carreguen al primer accés i es guarden en memòria
        self._voice_cache   = {}
        self._tampura_cache = {}

    # ------------------------------------------------------------------
    def _load(self, path: Path, cache: dict) -> np.ndarray:
        if path not in cache:
            audio, _ = librosa.load(path, sr=self.sr, mono=True)
            cache[path] = audio
        return cache[path]

    def _stft_mag_phase(self, audio: np.ndarray):
        D = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        return np.abs(D), np.angle(D)

    def _random_patch(self, mag: np.ndarray) -> tuple[int, int]:
        """Retorna (inici, fi) d'un patch aleatori en l'eix temporal."""
        T = mag.shape[1]
        if T <= self.patch_frames:
            return 0, T
        start = random.randint(0, T - self.patch_frames)
        return start, start + self.patch_frames

    def _pad_or_trim(self, mag: np.ndarray, t0: int, t1: int) -> np.ndarray:
        patch = mag[:, t0:t1]
        if patch.shape[1] < self.patch_frames:
            pad = self.patch_frames - patch.shape[1]
            patch = np.pad(patch, ((0, 0), (0, pad)))
        return patch

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        # Cada parell (voice, tampura) és un exemple; fem ×10 per diversitat de patches
        return len(self.voice_files) * len(self.tampura_files) * 10

    def __getitem__(self, idx):
        v_idx  = (idx // 10) % len(self.voice_files)
        t_idx  = (idx // 10) // len(self.voice_files) % len(self.tampura_files)

        voice   = self._load(self.voice_files[v_idx],   self._voice_cache).copy()
        tampura = self._load(self.tampura_files[t_idx], self._tampura_cache).copy()

        # Alinea longituds: retalla o fa loop
        min_len = min(len(voice), len(tampura))
        if len(tampura) < len(voice):
            reps = int(np.ceil(len(voice) / len(tampura)))
            tampura = np.tile(tampura, reps)
        voice   = voice[:min_len]
        tampura = tampura[:min_len]

        # Normalitza amplituds
        voice   = voice   / (np.abs(voice).max()   + 1e-8)
        tampura = tampura / (np.abs(tampura).max() + 1e-8)

        # Mescla sintètica
        alpha = random.uniform(*self.alpha_range)
        mix   = voice + alpha * tampura

        # STFT
        mag_mix,   _           = self._stft_mag_phase(mix)
        mag_voice, phase_mix   = self._stft_mag_phase(voice)
        # Nota: la fase del mix és la que s'usa a inferència per reconstruir l'àudio

        # Patch aleatori
        t0, t1 = self._random_patch(mag_mix)
        mag_mix_p   = self._pad_or_trim(mag_mix,   t0, t1)
        mag_voice_p = self._pad_or_trim(mag_voice, t0, t1)
        phase_p     = self._pad_or_trim(
            np.angle(librosa.stft(mix, n_fft=self.n_fft, hop_length=self.hop_length)),
            t0, t1,
        )

        # Normalitza magnituds al rang [0, 1] (per patch)
        norm = mag_mix_p.max() + 1e-8
        mag_mix_p   = mag_mix_p   / norm
        mag_voice_p = mag_voice_p / norm

        return (
            torch.tensor(mag_mix_p[np.newaxis],   dtype=torch.float32),  # (1, F, T)
            torch.tensor(mag_voice_p[np.newaxis],  dtype=torch.float32),  # (1, F, T)
            torch.tensor(phase_p[np.newaxis],      dtype=torch.float32),  # (1, F, T)
        )
