"""
U-Net per separació de tampura, adaptat de:
  Jansson et al. "Singing Voice Separation with Deep U-Net Convolutional
  Networks", ISMIR 2017.

Diferències respecte al paper originals:
  - Kernels 3×3 en lloc de 5×5 (versió petita, més eficient)
  - MaxPool2d + ConvTranspose2d en lloc de strided conv/deconv
    (funcionalment equivalent)
  - Profunditat 3 nivells + bottleneck en lloc de 6 (versió petita)

Fidel al paper en:
  - LeakyReLU(0.2) a l'encoder
  - ReLU al decoder
  - Dropout(0.5) als 3 primers blocs del decoder
  - Skip connections per concatenació
  - Sigmoid a la sortida (màscara suau)
  - La màscara es multiplica per l'espectre mixt fora del model

Ús previst:
  - Dades: veu_neta + α * tampura → mix
  - Entrada del model: |STFT(mix)|  shape (B, 1, F, T)
  - Sortida: màscara  shape (B, 1, F, T)
  - Veu reconstruïda: mask * |STFT(mix)|, fase de mix
"""

import torch
import torch.nn as nn


def _enc_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=3, padding=1),
        nn.BatchNorm2d(cout),
        nn.LeakyReLU(0.2, inplace=True),   # paper: leakiness 0.2
        nn.Conv2d(cout, cout, kernel_size=3, padding=1),
        nn.BatchNorm2d(cout),
        nn.LeakyReLU(0.2, inplace=True),
    )


def _dec_block(cin: int, cout: int, dropout: bool = False) -> nn.Sequential:
    layers = [
        nn.Conv2d(cin, cout, kernel_size=3, padding=1),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, kernel_size=3, padding=1),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    ]
    if dropout:
        layers.append(nn.Dropout2d(0.5))   # paper: 50% dropout als 3 primers decoder
    return nn.Sequential(*layers)


class UNetSmall(nn.Module):
    """
    U-Net petit (3 nivells + bottleneck) per separació de tampura.

    Args:
        in_channels : canals d'entrada (1 per espectrograma mono)
        base        : filtres base; els nivells fan base, base*2, base*4, base*8
    """

    def __init__(self, in_channels: int = 1, base: int = 32):
        super().__init__()

        self.enc1 = _enc_block(in_channels, base)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = _enc_block(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = _enc_block(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = _enc_block(base * 4, base * 8)

        # decoder: dropout als 3 primers (= tots en aquesta versió de 3 nivells)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec3 = _dec_block(base * 8, base * 4, dropout=True)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = _dec_block(base * 4, base * 2, dropout=True)

        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = _dec_block(base * 2, base, dropout=True)

        self.out = nn.Sequential(
            nn.Conv2d(base, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : espectrograma de magnitud  (B, 1, F, T)
        Returns:
            mask : màscara suau en [0, 1]  (B, 1, F, T)
        """
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        b = self.bottleneck(self.pool3(e3))

        d3 = self.dec3(torch.cat([self.up3(b),  e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out(d1)

    def separate(
        self,
        mag_mix: torch.Tensor,
        phase_mix: torch.Tensor,
        n_fft: int = 1024,
        hop_length: int = 256,
        length: int | None = None,
    ) -> torch.Tensor:
        """
        Aplica la màscara i reconstrueix l'àudio.

        Args:
            mag_mix   : magnitud STFT del mix     (B, 1, F, T)
            phase_mix : fase STFT del mix          (B, 1, F, T)
            length    : longitud original en mostres (per istft)
        Returns:
            àudio reconstruït  (B, samples)
        """
        mask = self.forward(mag_mix)
        mag_voice = mask * mag_mix
        stft_voice = mag_voice * torch.exp(1j * phase_mix)  # (B, 1, F, T) complex

        # ISTFT per cada element del batch
        waves = []
        for i in range(stft_voice.shape[0]):
            w = torch.istft(
                stft_voice[i, 0],
                n_fft=n_fft,
                hop_length=hop_length,
                length=length,
            )
            waves.append(w)
        return torch.stack(waves)
