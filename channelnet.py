"""
channelnet.py  —  ChannelNet implementation
"Deep Learning-Based Channel Estimation" (Soltani et al., IEEE Comms Letters 2019)

Architecture:
  Super-Resolution Network (SRNet) + Denoising CNN (DnCNN)
  Takes LS-interpolated channel as input, outputs refined estimate
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Super Resolution Network (SRNet)
# ---------------------------------------------------------------------------

class SRNet(nn.Module):
    """
    Upsampling / super-resolution branch of ChannelNet.
    Learns to refine the bilinear interpolated channel.

    Input  : (B, 2, Nf, Nt)  — real + imag as 2 channels
    Output : (B, 2, Nf, Nt)
    """

    def __init__(self, n_filters: int = 64, n_layers: int = 8):
        super().__init__()

        layers = []

        # First layer — no BN
        layers += [
            nn.Conv2d(2, n_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]

        # Middle layers — with BN
        for _ in range(n_layers - 2):
            layers += [
                nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1),
                nn.BatchNorm2d(n_filters),
                nn.ReLU(inplace=True),
            ]

        # Last layer — no BN, no activation
        layers += [
            nn.Conv2d(n_filters, 2, kernel_size=3, padding=1),
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.net(x)   # residual connection


# ---------------------------------------------------------------------------
# Denoising CNN (DnCNN)
# ---------------------------------------------------------------------------

class DnCNN(nn.Module):
    """
    Denoising branch of ChannelNet.
    Removes noise from SR output.

    Input  : (B, 2, Nf, Nt)
    Output : (B, 2, Nf, Nt)
    """

    def __init__(self, n_filters: int = 64, n_layers: int = 6):
        super().__init__()

        layers = []

        # First layer — no BN
        layers += [
            nn.Conv2d(2, n_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]

        # Middle layers — with BN
        for _ in range(n_layers - 2):
            layers += [
                nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1),
                nn.BatchNorm2d(n_filters),
                nn.ReLU(inplace=True),
            ]

        # Last layer
        layers += [
            nn.Conv2d(n_filters, 2, kernel_size=3, padding=1),
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return x - self.net(x)   # residual noise subtraction


# ---------------------------------------------------------------------------
# ChannelNet — SRNet + DnCNN combined
# ---------------------------------------------------------------------------

class ChannelNet(nn.Module):
    """
    Full ChannelNet pipeline:
      1. LS estimation at pilots
      2. Bilinear interpolation
      3. Stack real/imag → (B, 2, Nf, Nt)
      4. SRNet  → refined channel
      5. DnCNN  → denoised channel

    Parameters
    ----------
    Nf          : number of subcarriers (default 120)
    Nt          : number of OFDM symbols (default 14)
    sr_filters  : filters in SRNet (default 64)
    sr_layers   : layers  in SRNet (default 8)
    dn_filters  : filters in DnCNN (default 64)
    dn_layers   : layers  in DnCNN (default 6)
    """

    def __init__(
        self,
        Nf: int = 120,
        Nt: int = 14,
        sr_filters: int = 64,
        sr_layers:  int = 8,
        dn_filters: int = 64,
        dn_layers:  int = 6,
    ):
        super().__init__()
        self.Nf = Nf
        self.Nt = Nt

        self.srnet = SRNet(n_filters=sr_filters, n_layers=sr_layers)
        self.dncnn = DnCNN(n_filters=dn_filters, n_layers=dn_layers)

    # ------------------------------------------------------------------
    # Shared utility (same as CE-ViT)
    # ------------------------------------------------------------------

    @staticmethod
    def bilinear_interpolate(H_ls_p, Nf, Nt):
        """
        H_ls_p : (B, Nf_p, Nt_p) complex
        returns: (B, Nf, Nt) complex
        """
        real = H_ls_p.real.unsqueeze(1)
        imag = H_ls_p.imag.unsqueeze(1)
        real_up = F.interpolate(real, size=(Nf, Nt),
                                mode='bilinear', align_corners=False)
        imag_up = F.interpolate(imag, size=(Nf, Nt),
                                mode='bilinear', align_corners=False)
        return torch.complex(real_up.squeeze(1), imag_up.squeeze(1))

    @staticmethod
    def to_2ch(H):
        """(B, Nf, Nt) complex → (B, 2, Nf, Nt) float"""
        return torch.stack([H.real, H.imag], dim=1)

    @staticmethod
    def from_2ch(x):
        """(B, 2, Nf, Nt) float → (B, Nf, Nt) complex"""
        return torch.complex(x[:, 0], x[:, 1])

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, H_ls_p, snr=None, doppler=None, delay=None):
        """
        H_ls_p : (B, Nf_p, Nt_p) complex
        snr, doppler, delay : unused — kept for API compatibility with
                              train.py and evaluate.py
        returns H_hat : (B, Nf, Nt) complex
        """
        # 1. Bilinear interpolation
        H_int = self.bilinear_interpolate(H_ls_p, self.Nf, self.Nt)

        # 2. Stack real/imag → (B, 2, Nf, Nt)
        x = self.to_2ch(H_int)

        # 3. Super-resolution
        x = self.srnet(x)

        # 4. Denoising
        x = self.dncnn(x)

        # 5. Back to complex
        return self.from_2ch(x)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_channelnet(**kwargs):
    return ChannelNet(**kwargs)