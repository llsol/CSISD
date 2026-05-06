"""
Spectrogram-Channels U-Net for voice + tanpura separation.

Based on:
  Oh et al. "Spectrogram-Channels U-Net: A Source Separation Model
  Viewing Each Channel as the Spectrogram of Each Source", 2018.

Differences from the paper:
  - 3 levels + bottleneck instead of 4 (small version)
  - F.interpolate for upsampling (avoids size mismatch with odd spatial dims)

Faithful to the paper:
  - ReLU in encoder and decoder (not LeakyReLU)
  - Dropout(0.4) in decoder upsampling blocks
  - 2-channel direct spectrogram output (voice + tanpura) with ReLU — no mask
  - Weighted L1 loss with alpha computed from real source volumes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _enc_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=3, padding=1),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, kernel_size=3, padding=1),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


def _dec_block(cin: int, cout: int) -> nn.Sequential:
    """Post-upsampling refinement block (3x3, no dropout)."""
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=3, padding=1),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, kernel_size=3, padding=1),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class UNetSmall(nn.Module):
    """
    Spectrogram-Channels U-Net (3 levels + bottleneck).

    Args:
        in_channels  : input channels (1 for mono spectrogram)
        out_channels : output channels (2 = voice + tanpura)
        base         : base filter count; levels use base, x2, x4, x8
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 2, base: int = 32):
        super().__init__()

        self.enc1 = _enc_block(in_channels, base)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = _enc_block(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = _enc_block(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = _enc_block(base * 4, base * 8)

        # Upsampling blocks: ConvTranspose2d to halve channels + Dropout(0.4)
        self.up3   = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
        self.drop3 = nn.Dropout2d(0.4)
        self.dec3  = _dec_block(base * 8, base * 4)

        self.up2   = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.drop2 = nn.Dropout2d(0.4)
        self.dec2  = _dec_block(base * 4, base * 2)

        self.up1   = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.drop1 = nn.Dropout2d(0.4)
        self.dec1  = _dec_block(base * 2, base)

        # Direct spectrogram output per source (not a mask)
        self.out = nn.Sequential(
            nn.Conv2d(base, out_channels, kernel_size=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x   : mix magnitude spectrogram  (B, 1, F, T)
        Returns:
            out : (B, 2, F, T)  — channel 0: voice, channel 1: tanpura
        """
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b  = self.bottleneck(self.pool3(e3))

        # F.interpolate corrects the 1-pixel mismatch from odd spatial dims (e.g. F=513)
        up3 = self.drop3(F.interpolate(self.up3(b),  size=e3.shape[2:], mode='nearest'))
        d3  = self.dec3(torch.cat([up3, e3], dim=1))

        up2 = self.drop2(F.interpolate(self.up2(d3), size=e2.shape[2:], mode='nearest'))
        d2  = self.dec2(torch.cat([up2, e2], dim=1))

        up1 = self.drop1(F.interpolate(self.up1(d2), size=e1.shape[2:], mode='nearest'))
        d1  = self.dec1(torch.cat([up1, e1], dim=1))

        return self.out(d1)
