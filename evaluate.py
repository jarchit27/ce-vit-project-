"""
evaluate.py  —  Reproduce CE-ViT paper figures (Figs 4-7)
Compares CE-ViT, ViT, and ChannelNet
"""

import math
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from ce_vit      import build_ce_vit, build_vit
from channel_sim import OFDMDataset, generate_tdl_channel, get_pilot_indices
from channelnet  import build_channelnet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_mse_db(H_hat, H_true):
    err = H_hat - H_true
    mse = (err.real ** 2 + err.imag ** 2).mean().item()
    return 10 * math.log10(mse + 1e-15)


@torch.no_grad()
def evaluate_model(model, dataset, device, use_tokens, batch_size=512):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    preds, trues = [], []
    for H_ls_p, H_true, snr, doppler, delay in loader:
        H_ls_p  = H_ls_p.to(device)
        snr     = snr.to(device)
        doppler = doppler.to(device)
        delay   = delay.to(device)

        if use_tokens:
            H_hat = model(H_ls_p, snr, doppler, delay)
        else:
            H_hat = model(H_ls_p)

        preds.append(H_hat.cpu())
        trues.append(H_true)

    preds = torch.cat(preds)
    trues = torch.cat(trues)
    return compute_mse_db(preds, trues)


# ---------------------------------------------------------------------------
# LS Baseline
# ---------------------------------------------------------------------------

def ls_mse_db(dataset):
    from ce_vit import CEViT
    tmp = CEViT()
    preds, trues = [], []
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    for H_ls_p, H_true, *_ in loader:
        H_int = tmp.bilinear_interpolate(H_ls_p, tmp.Nf, tmp.Nt)
        preds.append(H_int)
        trues.append(H_true)
    preds = torch.cat(preds)
    trues = torch.cat(trues)
    return compute_mse_db(preds, trues)


# ---------------------------------------------------------------------------
# Sweep MSE vs one parameter
# ---------------------------------------------------------------------------

def sweep_mse(
    model,
    device,
    use_tokens,
    sweep_param: str,
    sweep_values: list,
    fixed: dict,
    n_samples_per_point: int = 2000,
    channel: str = "TDL-A",
):
    mse_list = []
    for val in sweep_values:
        kw = dict(fixed)
        kw[sweep_param] = val
        ds = OFDMDataset(
            n_samples=n_samples_per_point,
            fixed_snr     = kw.get('snr'),
            fixed_doppler = kw.get('doppler'),
            fixed_delay   = kw.get('delay'),
            channel=channel,
            seed=999,
        )
        mse = evaluate_model(model, ds, device, use_tokens)
        mse_list.append(mse)
        print(f"  {sweep_param}={val:8g}  MSE={mse:.2f} dB")
    return mse_list


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load CE-ViT ----
    ce_vit = build_ce_vit(d_model=128, n_heads=4, n_layers=4).to(device)
    if args.ckpt:
        ce_vit.load_state_dict(torch.load(args.ckpt, map_location=device))
        print(f"Loaded CE-ViT: {args.ckpt}")
    ce_vit.eval()

    # ---- Load ViT ----
    vit = build_vit(d_model=128, n_heads=4, n_layers=4).to(device)
    if args.ckpt_vit:
        vit.load_state_dict(torch.load(args.ckpt_vit, map_location=device))
        print(f"Loaded ViT: {args.ckpt_vit}")
    vit.eval()

    # ---- Load ChannelNet ----
    channelnet = build_channelnet().to(device)
    if args.ckpt_channelnet:
        channelnet.load_state_dict(torch.load(args.ckpt_channelnet,
                                              map_location=device))
        print(f"Loaded ChannelNet: {args.ckpt_channelnet}")
    channelnet.eval()

    results = {}
    snr_vals   = list(range(0, 31, 5))
    dop_vals   = list(range(200, 1700, 200))
    delay_vals = list(range(50, 450, 50))

    # ---- Fig 4: MSE vs SNR ----
    print("\n[Fig 4] MSE vs SNR  (Doppler=500Hz, Delay=200ns)")
    r_ce = sweep_mse(ce_vit,     device, True,  'snr', snr_vals, {'doppler': 500, 'delay': 200})
    r_vit= sweep_mse(vit,        device, False, 'snr', snr_vals, {'doppler': 500, 'delay': 200})
    r_cn = sweep_mse(channelnet, device, False, 'snr', snr_vals, {'doppler': 500, 'delay': 200})
    results['fig4'] = {'snr': snr_vals, 'CE-ViT': r_ce, 'ViT': r_vit, 'ChannelNet': r_cn}

    # ---- Fig 5: MSE vs Doppler ----
    print("\n[Fig 5] MSE vs Doppler  (Delay=200ns, SNR=20dB)")
    r_ce = sweep_mse(ce_vit,     device, True,  'doppler', dop_vals, {'snr': 20, 'delay': 200})
    r_vit= sweep_mse(vit,        device, False, 'doppler', dop_vals, {'snr': 20, 'delay': 200})
    r_cn = sweep_mse(channelnet, device, False, 'doppler', dop_vals, {'snr': 20, 'delay': 200})
    results['fig5'] = {'doppler': dop_vals, 'CE-ViT': r_ce, 'ViT': r_vit, 'ChannelNet': r_cn}

    # ---- Fig 6: MSE vs Delay ----
    print("\n[Fig 6] MSE vs Delay  (Doppler=500Hz, SNR=20dB)")
    r_ce = sweep_mse(ce_vit,     device, True,  'delay', delay_vals, {'snr': 20, 'doppler': 500})
    r_vit= sweep_mse(vit,        device, False, 'delay', delay_vals, {'snr': 20, 'doppler': 500})
    r_cn = sweep_mse(channelnet, device, False, 'delay', delay_vals, {'snr': 20, 'doppler': 500})
    results['fig6'] = {'delay': delay_vals, 'CE-ViT': r_ce, 'ViT': r_vit, 'ChannelNet': r_cn}

    # ---- Fig 7: Robustness ----
    print("\n[Fig 7] Robustness  (TDL-E, Delay=400ns, Doppler=1200Hz)")
    r_ce = sweep_mse(ce_vit,     device, True,  'snr', snr_vals, {'doppler': 1200, 'delay': 400}, channel='TDL-E')
    r_vit= sweep_mse(vit,        device, False, 'snr', snr_vals, {'doppler': 1200, 'delay': 400}, channel='TDL-E')
    r_cn = sweep_mse(channelnet, device, False, 'snr', snr_vals, {'doppler': 1200, 'delay': 400}, channel='TDL-E')
    results['fig7'] = {'snr': snr_vals, 'CE-ViT': r_ce, 'ViT': r_vit, 'ChannelNet': r_cn}

    # ---- Print summary table ----
    MODEL_KEYS = ('CE-ViT', 'ViT', 'ChannelNet')
    print("\n===== MSE Summary (dB) =====")
    for fig, data in results.items():
        x_key = [k for k in data if k not in MODEL_KEYS][0]
        print(f"\n{fig.upper()}  ({x_key})")
        print(f"{'x':>12}  {'CE-ViT':>10}  {'ViT':>10}  {'ChannelNet':>12}")
        for i, x in enumerate(data[x_key]):
            print(f"{x:12g}  {data['CE-ViT'][i]:10.2f}  "
                  f"{data['ViT'][i]:10.2f}  {data['ChannelNet'][i]:12.2f}")

    # ---- Plot ----
    if args.plot:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle("CE-ViT vs ViT vs ChannelNet (Reproduced)")

            def plot_ax(ax, x, data, x_key, xlabel, title):
                markers = {'CE-ViT': 'o', 'ViT': 's', 'ChannelNet': '^'}
                for label in MODEL_KEYS:
                    if label in data:
                        ax.plot(x, data[label], marker=markers[label],
                                label=label)
                ax.set_xlabel(xlabel)
                ax.set_ylabel("MSE (dB)")
                ax.set_title(title)
                ax.legend()
                ax.grid(True)

            d = results['fig4']
            plot_ax(axes[0,0], d['snr'],     d, 'snr',
                    "SNR (dB)",          "MSE vs SNR")

            d = results['fig5']
            plot_ax(axes[0,1], d['doppler'], d, 'doppler',
                    "Max Doppler (Hz)",  "MSE vs Doppler")

            d = results['fig6']
            plot_ax(axes[1,0], d['delay'],   d, 'delay',
                    "Delay spread (ns)", "MSE vs Delay")

            d = results['fig7']
            plot_ax(axes[1,1], d['snr'],     d, 'snr',
                    "SNR (dB)",          "Robustness (TDL-E mismatch)")

            plt.tight_layout()
            plt.savefig("ce_vit_results.png", dpi=150)
            print("\nPlot saved to ce_vit_results.png")

        except ImportError:
            print("matplotlib not available — skipping plots")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",             type=str, default=None)
    parser.add_argument("--ckpt_vit",         type=str, default=None)
    parser.add_argument("--ckpt_channelnet",  type=str, default=None)
    parser.add_argument("--plot",             action="store_true")
    args = parser.parse_args()
    main(args)