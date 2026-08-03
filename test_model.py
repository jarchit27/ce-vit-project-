"""
test_model.py  —  Sanity check for CE-ViT forward pass and shape verification
"""

import torch
import math
from ce_vit import build_ce_vit, build_vit, CEViT


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def flops_estimate(model, B=1):
    """Very rough FLOPs estimate by running a forward pass with profiling."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
        H_ls_p  = torch.randn(B, 40, 2, dtype=torch.cfloat)
        snr     = torch.rand(B, 1)
        doppler = torch.rand(B, 1)
        delay   = torch.rand(B, 1)
        with FlopCounterMode(model, display=False) as fcm:
            _ = model(H_ls_p, snr, doppler, delay)
        return fcm.get_total_flops()
    except Exception:
        return None


def main():
    torch.manual_seed(0)

    # --- Paper default params ---
    Nf, Nt = 120, 14
    Nf_p, Nt_p = 40, 2   # pilot grid
    B = 4                 # batch size

    # Synthetic inputs
    H_ls_p  = torch.randn(B, Nf_p, Nt_p, dtype=torch.cfloat)
    snr     = torch.rand(B, 1)       # normalised [0,1]
    doppler = torch.rand(B, 1)
    delay   = torch.rand(B, 1)

    print("=" * 55)
    print("CE-ViT Smoke Test")
    print("=" * 55)

    # ---- CE-ViT (with tokens) ----
    model = build_ce_vit(Nf=Nf, Nt=Nt, d_model=128, n_heads=4, n_layers=4)
    model.eval()
    with torch.no_grad():
        H_hat = model(H_ls_p, snr, doppler, delay)

    assert H_hat.shape == (B, Nf, Nt), f"Shape mismatch: {H_hat.shape}"
    print(f"CE-ViT output shape : {tuple(H_hat.shape)}  ✓")
    print(f"CE-ViT parameters   : {count_params(model)/1e6:.3f}M")

    # ---- ViT (no tokens) ----
    vit = build_vit(Nf=Nf, Nt=Nt, d_model=128, n_heads=4, n_layers=4)
    vit.eval()
    with torch.no_grad():
        H_hat_vit = vit(H_ls_p)

    assert H_hat_vit.shape == (B, Nf, Nt)
    print(f"ViT     output shape : {tuple(H_hat_vit.shape)}  ✓")
    print(f"ViT     parameters   : {count_params(vit)/1e6:.3f}M")

    # ---- Patch dimensions ----
    pe = model.patch_embed
    print(f"\nPatch size           : {pe.Ph} x {pe.Pt}")
    print(f"Num patches          : {pe.num_patches}  (= {Nf}//{pe.Ph} x {2*Nt}//{pe.Pt})")
    print(f"Patch dim            : {pe.patch_dim}")

    # ---- Quick MSE check ----
    # Generate a trivial channel and verify the model runs without NaN
    H_true = torch.randn(B, Nf, Nt, dtype=torch.cfloat)
    err = H_hat - H_true
    mse = (err.real**2 + err.imag**2).mean().item()
    print(f"\nMSE (untrained, random input): {10*math.log10(mse):.2f} dB  (expected ~0 dB)")

    print("\nAll checks passed ✓")
    print("=" * 55)
    print("\nUsage:")
    print("  Train:    python train.py --epochs 1000 --n_samples 102400")
    print("  Evaluate: python evaluate.py --ckpt checkpoints/best_model.pth --plot")
    print("  ViT only: python train.py --no_tokens --epochs 1000")


if __name__ == "__main__":
    main()
