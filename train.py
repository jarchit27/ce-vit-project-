"""
train.py  —  Train CE-ViT / ViT / ChannelNet
"""

import argparse
import math
import os
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split

from ce_vit      import build_ce_vit, build_vit
from channelnet  import build_channelnet
from channel_sim import OFDMDataset


def mse_db(pred, target):
    err = pred - target
    mse = (err.real ** 2 + err.imag ** 2).mean()
    return 10 * math.log10(mse.item() + 1e-15)


def train_epoch(model, loader, criterion, optimizer, device, use_tokens):
    model.train()
    total_loss = 0.0
    for H_ls_p, H_true, snr, doppler, delay in loader:
        H_ls_p  = H_ls_p.to(device)
        H_true  = H_true.to(device)
        snr     = snr.to(device)
        doppler = doppler.to(device)
        delay   = delay.to(device)

        if use_tokens:
            H_hat = model(H_ls_p, snr, doppler, delay)
        else:
            H_hat = model(H_ls_p)

        loss = (
            criterion(H_hat.real, H_true.real) +
            criterion(H_hat.imag, H_true.imag)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, use_tokens):
    model.eval()
    total_loss = 0.0
    for H_ls_p, H_true, snr, doppler, delay in loader:
        H_ls_p  = H_ls_p.to(device)
        H_true  = H_true.to(device)
        snr     = snr.to(device)
        doppler = doppler.to(device)
        delay   = delay.to(device)

        if use_tokens:
            H_hat = model(H_ls_p, snr, doppler, delay)
        else:
            H_hat = model(H_ls_p)

        loss = (
            criterion(H_hat.real, H_true.real) +
            criterion(H_hat.imag, H_true.imag)
        )
        total_loss += loss.item()

    return total_loss / len(loader)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Dataset ----
    print("Generating training dataset ...")
    full_dataset = OFDMDataset(n_samples=args.n_samples, seed=42)
    n_val   = int(0.10 * len(full_dataset))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(0)
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)

    # ---- Model ----
    if args.model == 'ce_vit':
        use_tokens = True
        model = build_ce_vit(d_model=args.d_model, n_heads=args.n_heads,
                             n_layers=args.n_layers)
        print("Model: CE-ViT (with tokens)")

    elif args.model == 'vit':
        use_tokens = False
        model = build_vit(d_model=args.d_model, n_heads=args.n_heads,
                          n_layers=args.n_layers)
        print("Model: ViT (no tokens)")

    elif args.model == 'channelnet':
        use_tokens = False
        model = build_channelnet()
        print("Model: ChannelNet")

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params/1e6:.3f}M")

    # ---- Optimizer + Scheduler ----
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=500, gamma=0.1)
    criterion = nn.MSELoss()

    # ---- Training Loop ----
    best_val = float("inf")
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, criterion,
                                 optimizer, device, use_tokens)
        val_loss   = eval_epoch(model, val_loader, criterion,
                                device, use_tokens)
        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"Epoch [{epoch:4d}/{args.epochs}]  "
            f"Train: {train_loss:.6f}  "
            f"Val: {val_loss:.6f}  "
            f"LR: {scheduler.get_last_lr()[0]:.2e}  "
            f"Time: {elapsed:.1f}s"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(),
                       os.path.join(args.save_dir, "best_model.pth"))

    print(f"\nDone. Best val loss: {best_val:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      type=str,   default="ce_vit",
                        choices=["ce_vit", "vit", "channelnet"])
    parser.add_argument("--n_samples",  type=int,   default=102400)
    parser.add_argument("--batch_size", type=int,   default=512)
    parser.add_argument("--epochs",     type=int,   default=1000)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--d_model",    type=int,   default=128)
    parser.add_argument("--n_heads",    type=int,   default=4)
    parser.add_argument("--n_layers",   type=int,   default=4)
    parser.add_argument("--save_dir",   type=str,   default="checkpoints")
    args = parser.parse_args()
    main(args)