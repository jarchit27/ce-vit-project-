"""
Plot true vs LS-estimated pilot values from OFDMDataset samples.

The dataset stores:
  H_ls_p : noisy LS estimates on the pilot grid, shape (40, 2)
  H_true : full true channel, shape (120, 14)

This script extracts H_true at the same pilot positions and saves side-by-side
image samples for magnitude, real part, and imaginary part.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from channel_sim import OFDMDataset, get_pilot_indices


def as_numpy(x):
    return x.detach().cpu().numpy()


def mse_complex(a, b):
    err = a - b
    return float(np.mean(err.real ** 2 + err.imag ** 2))


def plot_one_sample(sample_idx, H_true_p, H_ls_p, out_dir):
    views = [
        ("Magnitude", np.abs(H_true_p), np.abs(H_ls_p), np.abs(H_ls_p - H_true_p)),
        ("Real", H_true_p.real, H_ls_p.real, (H_ls_p - H_true_p).real),
        ("Imag", H_true_p.imag, H_ls_p.imag, (H_ls_p - H_true_p).imag),
    ]

    fig, axes = plt.subplots(len(views), 3, figsize=(11, 8), constrained_layout=True)
    fig.suptitle(
        f"Sample {sample_idx} pilot values | MSE={mse_complex(H_ls_p, H_true_p):.4e}",
        fontsize=12,
    )

    for row, (name, true_img, est_img, err_img) in enumerate(views):
        panels = [
            (f"True {name}", true_img),
            (f"LS Estimated {name}", est_img),
            (f"Error {name}", err_img),
        ]

        for col, (title, data) in enumerate(panels):
            ax = axes[row, col]
            im = ax.imshow(data, aspect="auto", origin="lower", cmap="viridis")
            ax.set_title(title)
            ax.set_xlabel("Pilot OFDM symbol index")
            ax.set_ylabel("Pilot subcarrier index")
            ax.set_xticks(range(data.shape[1]))
            fig.colorbar(im, ax=ax, shrink=0.85)

    out_path = out_dir / f"pilot_sample_{sample_idx:02d}.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Save true-vs-estimated pilot value image samples."
    )
    parser.add_argument("--n_samples", type=int, default=6)
    parser.add_argument("--Nf", type=int, default=120)
    parser.add_argument("--Nt", type=int, default=14)
    parser.add_argument("--channel", type=str, default="TDL-A", choices=["TDL-A", "TDL-E"])
    parser.add_argument("--snr", type=float, default=20.0)
    parser.add_argument("--doppler", type=float, default=100.0)
    parser.add_argument("--delay", type=float, default=500.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="pilot_sample_plots")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = OFDMDataset(
        n_samples=args.n_samples,
        Nf=args.Nf,
        Nt=args.Nt,
        channel=args.channel,
        fixed_snr=args.snr,
        fixed_doppler=args.doppler,
        fixed_delay=args.delay,
        seed=args.seed,
    )
    freq_idx, time_idx = get_pilot_indices(args.Nf, args.Nt)

    print(f"Saving {args.n_samples} pilot comparison plots to {out_dir.resolve()}")
    for i in range(args.n_samples):
        H_ls_p, H_true, snr, doppler, delay = dataset[i]
        H_ls_p = as_numpy(H_ls_p)
        H_true = as_numpy(H_true)
        H_true_p = H_true[np.ix_(freq_idx, time_idx)]

        out_path = plot_one_sample(i, H_true_p, H_ls_p, out_dir)
        print(
            f"{out_path} | H_ls_p={H_ls_p.shape}, H_true_p={H_true_p.shape}, "
            f"MSE={mse_complex(H_ls_p, H_true_p):.4e}, "
            f"snr={float(snr.item()):.3f}, doppler={float(doppler.item()):.3f}, "
            f"delay={float(delay.item()):.3f}"
        )


if __name__ == "__main__":
    main()
