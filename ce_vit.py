"""
CE-ViT: A Robust Channel Estimator Based on Vision Transformer for OFDM Systems
GLOBECOM 2023

Architecture:
  1. Upsampling Module     - LS estimation + bilinear interpolation
  2. Patch Embedding       - 10x4 patches flattened to 1D
  3. Token Module          - SNR / Doppler / Delay tokens via FC layers
  4. Transformer Encoder   - Multi-head attention + FFN
  5. Inverse Patch Embedding - reconstruct channel estimate
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Patch Embedding / Inverse Patch Embedding
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Partition H_int (Nf x 2*Nt real-valued) into (Ph x Pt) patches and
    flatten each patch to a 1-D token.

    Input shape:  (B, Nf, 2*Nt)   -- real + imag stacked along time axis
    Output shape: (B, num_patches, patch_dim)
                  where num_patches = (Nf/Ph) * (2*Nt/Pt)
                        patch_dim   = Ph * Pt
    """

    def __init__(self, Nf: int, Nt: int, Ph: int = 10, Pt: int = 4):
        super().__init__()
        assert Nf % Ph == 0 and (2 * Nt) % Pt == 0, \
            "Grid dimensions must be divisible by patch sizes"
        self.Nf, self.Nt = Nf, Nt
        self.Ph, self.Pt = Ph, Pt
        self.nph = Nf // Ph          # patches along freq
        self.npt = (2 * Nt) // Pt   # patches along time (real+imag)
        self.num_patches = self.nph * self.npt
        self.patch_dim   = Ph * Pt

    def forward(self, x):
        # x: (B, Nf, 2*Nt)
        B, Nf, T = x.shape           # T = 2*Nt
        # reshape to (B, nph, Ph, npt, Pt)
        x = x.view(B, self.nph, self.Ph, self.npt, self.Pt)
        # -> (B, nph, npt, Ph, Pt) -> (B, num_patches, patch_dim)
        x = x.permute(0, 1, 3, 2, 4).contiguous()
        x = x.view(B, self.num_patches, self.patch_dim)
        return x


class InversePatchEmbed(nn.Module):
    """Reconstruct the (Nf x 2*Nt) grid from patch tokens.

    Input shape:  (B, num_patches, patch_dim)
    Output shape: (B, Nf, 2*Nt)
    """

    def __init__(self, Nf: int, Nt: int, Ph: int = 10, Pt: int = 4):
        super().__init__()
        self.Nf, self.Nt = Nf, Nt
        self.Ph, self.Pt = Ph, Pt
        self.nph = Nf // Ph
        self.npt = (2 * Nt) // Pt

    def forward(self, x):
        B, num_patches, patch_dim = x.shape
        x = x.view(B, self.nph, self.npt, self.Ph, self.Pt)
        x = x.permute(0, 1, 3, 2, 4).contiguous()
        x = x.view(B, self.Nf, 2 * self.Nt)
        return x


# ---------------------------------------------------------------------------
# Token Module
# ---------------------------------------------------------------------------

class TokenModule(nn.Module):
    """Convert scalar channel parameters (SNR, Doppler, Delay) to tokens.

    Each scalar → FC → R^(num_patches/2)  → reshape to (num_patches/2 x 2)
    Final tokens concatenated along dim-1: (num_patches x 2)  [Delay, SNR, Doppler → 3 tokens of size (num_patches x 2)]
    Actually per paper: three FCs → tokens of size R^(NfNt/10)
    reshaped to (NfNt/20 x 2) each, giving 3 tokens → total shape (NfNt/20 x 6).
    We concatenate them as 6 extra "sequence" positions along the sequence dimension.
    """

    def __init__(self, num_patches: int, token_seq: int = 6):
        """
        num_patches : NfNt/20  (sequence length after patch embedding)
        token_seq   : 6 extra token columns appended to patch_dim
        The paper describes tokens reshaped to (NfNt/20 x 2) for each of
        the 3 parameters → concatenated with patched channel on dim-1 (feature dim)
        giving total feature size = patch_dim + 6 = 40 + 6 = 46.
        """
        super().__init__()
        self.token_dim = num_patches * 2   # NfNt/10  (paper notation)

        self.fc_snr     = nn.Linear(1, self.token_dim)
        self.fc_doppler = nn.Linear(1, self.token_dim)
        self.fc_delay   = nn.Linear(1, self.token_dim)

        self.num_patches = num_patches

    def forward(self, snr, doppler, delay):
        """
        snr, doppler, delay : (B, 1)
        returns token_concat : (B, num_patches, 6)
        """
        t_snr     = self.fc_snr(snr)                          # (B, num_patches*2)
        t_doppler = self.fc_doppler(doppler)
        t_delay   = self.fc_delay(delay)

        # reshape to (B, num_patches, 2)
        t_snr     = t_snr.view(-1, self.num_patches, 2)
        t_doppler = t_doppler.view(-1, self.num_patches, 2)
        t_delay   = t_delay.view(-1, self.num_patches, 2)

        tokens = torch.cat([t_snr, t_doppler, t_delay], dim=-1)  # (B, num_patches, 6)
        return tokens


# ---------------------------------------------------------------------------
# Transformer Encoder
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )

    def forward(self, x):
        return self.net(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = FeedForward(d_model)

    def forward(self, x):
        # Pre-norm + residual (as in ViT)
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# CE-ViT
# ---------------------------------------------------------------------------

class CEViT(nn.Module):
    """
    CE-ViT channel estimator for OFDM systems.

    Parameters
    ----------
    Nf       : number of subcarriers  (default 120)
    Nt       : number of OFDM symbols (default 14)
    Ph, Pt   : patch height / width   (default 10, 4)
    d_model  : transformer hidden dim (default 128)
    n_heads  : attention heads        (default 4)
    n_layers : transformer depth      (default 4)
    use_tokens : include token module (CE-ViT=True, ViT=False)
    """

    def __init__(
        self,
        Nf: int = 120,
        Nt: int = 14,
        Ph: int = 10,
        Pt: int = 4,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        use_tokens: bool = True,
    ):
        super().__init__()
        self.Nf, self.Nt = Nf, Nt
        self.use_tokens = use_tokens

        # ---- Patch Embedding ----
        self.patch_embed = PatchEmbed(Nf, Nt, Ph, Pt)
        num_patches = self.patch_embed.num_patches   # NfNt/20
        patch_dim   = self.patch_embed.patch_dim     # 40

        # ---- Token Module ----
        token_extra_dim = 6 if use_tokens else 0
        if use_tokens:
            self.token_module = TokenModule(num_patches)

        # ---- Input projection (patch_dim + tokens → d_model) ----
        in_dim = patch_dim + token_extra_dim
        self.input_proj = nn.Linear(in_dim, d_model)

        # ---- Learnable Positional Encoding ----
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # ---- Transformer Encoder ----
        self.encoder = nn.Sequential(
            *[TransformerEncoderBlock(d_model, n_heads) for _ in range(n_layers)]
        )

        # ---- Output projection (d_model → patch_dim) ----
        self.output_proj = nn.Linear(d_model, patch_dim)

        # ---- Inverse Patch Embedding ----
        self.inv_patch_embed = InversePatchEmbed(Nf, Nt, Ph, Pt)

    # ------------------------------------------------------------------
    # Upsampling Module (LS + bilinear interpolation)
    # ------------------------------------------------------------------
    @staticmethod
    def ls_estimate(y_pilot, x_pilot):
        """
        LS channel estimate at pilot positions.
        y_pilot, x_pilot : (B, Nf_p, Nt_p) complex tensors
        returns H_ls_p   : (B, Nf_p, Nt_p) complex
        """
        return y_pilot * (1.0 / x_pilot)

    @staticmethod
    def bilinear_interpolate(H_ls_p, Nf: int, Nt: int):
        """
        Bilinear interpolation of pilot-position LS estimates
        to full (Nf x Nt) grid.

        H_ls_p : (B, Nf_p, Nt_p) complex
        returns: (B, Nf, Nt) complex
        """
        B = H_ls_p.shape[0]
        # treat real/imag separately for F.interpolate
        real = H_ls_p.real.unsqueeze(1)  # (B,1,Nf_p,Nt_p)
        imag = H_ls_p.imag.unsqueeze(1)

        real_up = F.interpolate(real, size=(Nf, Nt), mode='bilinear', align_corners=False)
        imag_up = F.interpolate(imag, size=(Nf, Nt), mode='bilinear', align_corners=False)

        H_int = torch.complex(real_up.squeeze(1), imag_up.squeeze(1))  # (B,Nf,Nt)
        return H_int

    # ------------------------------------------------------------------
    # Real-valued stacking: concat real + imag along time dim
    # ------------------------------------------------------------------
    @staticmethod
    def to_real_stacked(H):
        """
        H : (B, Nf, Nt) complex
        returns (B, Nf, 2*Nt) float — real then imag halves
        """
        return torch.cat([H.real, H.imag], dim=-1)

    @staticmethod
    def from_real_stacked(x, Nt: int):
        """
        x  : (B, Nf, 2*Nt) float
        returns (B, Nf, Nt) complex
        """
        return torch.complex(x[..., :Nt], x[..., Nt:])

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, H_ls_p, snr=None, doppler=None, delay=None):
        """
        H_ls_p  : (B, Nf_p, Nt_p) complex  — LS at pilot positions
        snr     : (B, 1) float  [dB, normalised before passing]
        doppler : (B, 1) float  [Hz, normalised]
        delay   : (B, 1) float  [ns, normalised]

        returns H_hat : (B, Nf, Nt) complex
        """
        # 1. Bilinear interpolation → (B, Nf, Nt) complex
        H_int = self.bilinear_interpolate(H_ls_p, self.Nf, self.Nt)

        # 2. Stack real / imag → (B, Nf, 2*Nt) float
        x = self.to_real_stacked(H_int)

        # 3. Patch Embedding → (B, num_patches, patch_dim=40)
        x = self.patch_embed(x)

        # 4. Token Module → concatenate tokens along feature dim
        if self.use_tokens and snr is not None:
            tokens = self.token_module(snr, doppler, delay)  # (B, num_patches, 6)
            x = torch.cat([x, tokens], dim=-1)               # (B, num_patches, 46)

        # 5. Input projection + positional encoding
        x = self.input_proj(x)       # (B, num_patches, d_model)
        x = x + self.pos_embed

        # 6. Transformer Encoder
        x = self.encoder(x)          # (B, num_patches, d_model)

        # 7. Output projection → (B, num_patches, patch_dim=40)
        x = self.output_proj(x)

        # 8. Inverse Patch Embedding → (B, Nf, 2*Nt)
        x = self.inv_patch_embed(x)

        # 9. Unstack real / imag → (B, Nf, Nt) complex
        H_hat = self.from_real_stacked(x, self.Nt)
        return H_hat


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_ce_vit(**kwargs):
    """Build CE-ViT (with tokens)."""
    return CEViT(use_tokens=True, **kwargs)


def build_vit(**kwargs):
    """Build ViT ablation (no tokens)."""
    return CEViT(use_tokens=False, **kwargs)
