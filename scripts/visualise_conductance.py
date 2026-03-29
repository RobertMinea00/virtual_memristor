"""
Visualise conductance distributions and drift over time.

Usage:
    .venv/Scripts/python scripts/visualise_conductance.py --log-file logs/run.pt
    .venv/Scripts/python scripts/visualise_conductance.py --live  (requires a running model)
"""

import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import yaml

from memristor.device_model import MemristorDeviceModel
from network.classifier import MemristorClassifier


def plot_conductance_snapshot(model: MemristorClassifier, title: str = "") -> None:
    layers = model.memristor_layers()
    fig = plt.figure(figsize=(14, 4 * len(layers)))
    gs = gridspec.GridSpec(len(layers), 3)

    for i, layer in enumerate(layers):
        g_pos = layer.G_pos.detach().cpu().numpy().flatten()
        g_neg = layer.G_neg.detach().cpu().numpy().flatten()
        g_eff = (g_pos - g_neg)

        ax1 = fig.add_subplot(gs[i, 0])
        ax1.hist(g_pos, bins=50, color="steelblue", alpha=0.7, label="G+")
        ax1.hist(g_neg, bins=50, color="tomato", alpha=0.7, label="G-")
        ax1.set_title(f"Layer {i}: Raw conductances")
        ax1.set_xlabel("Conductance (S)")
        ax1.legend()

        ax2 = fig.add_subplot(gs[i, 1])
        ax2.hist(g_eff, bins=50, color="mediumseagreen", alpha=0.8)
        ax2.set_title(f"Layer {i}: Effective weight (G+ - G-)")
        ax2.set_xlabel("Effective weight")

        ax3 = fig.add_subplot(gs[i, 2])
        # Saturation map: fraction close to g_min or g_max
        with open("config/memristor.yaml") as f:
            cfg = yaml.safe_load(f)
        g_min, g_max = cfg["g_min"], cfg["g_max"]
        sat_high = (g_pos > 0.95 * g_max).mean() * 100
        sat_low  = (g_pos < g_min * 1.05).mean() * 100
        ax3.bar(["Saturated high", "Saturated low"], [sat_high, sat_low],
                color=["tomato", "steelblue"])
        ax3.set_title(f"Layer {i}: G+ saturation (%)")
        ax3.set_ylim(0, 100)

    plt.suptitle(title or "Conductance distribution snapshot", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_from_log(log_path: str) -> None:
    """
    Plot conductance statistics over training from a saved benchmark log.
    Expects list of per-step dicts with 'conductance' key.
    """
    data = torch.load(log_path, map_location="cpu")
    steps = [d["sample"] for d in data if "conductance" in d]
    cond_logs = [d["conductance"] for d in data if "conductance" in d]

    if not steps:
        print("No conductance data found in log.")
        return

    layer_keys = list(cond_logs[0].keys())
    metrics = ["g_eff_mean", "g_eff_std", "fraction_saturated"]

    fig, axes = plt.subplots(len(layer_keys), len(metrics),
                             figsize=(5 * len(metrics), 3 * len(layer_keys)))

    for li, lkey in enumerate(layer_keys):
        for mi, metric in enumerate(metrics):
            vals = [c[lkey][metric] for c in cond_logs]
            ax = axes[li][mi] if len(layer_keys) > 1 else axes[mi]
            ax.plot(steps, vals)
            ax.set_title(f"{lkey}: {metric}")
            ax.set_xlabel("Sample")

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--model-path", default=None)
    args = parser.parse_args()

    if args.log_file:
        plot_from_log(args.log_file)
    elif args.model_path:
        with open("config/network.yaml") as f:
            net_cfg = yaml.safe_load(f)
        with open("config/memristor.yaml") as f:
            mem_cfg = yaml.safe_load(f)
        dm = MemristorDeviceModel(mem_cfg)
        model = MemristorClassifier(
            n_classes=net_cfg["initial_classes"],
            input_dim=net_cfg["input_dim"],
            hidden_dims=net_cfg["hidden_dims"],
            device_model=dm,
        )
        model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
        plot_conductance_snapshot(model)
    else:
        print("Provide --log-file or --model-path")


if __name__ == "__main__":
    main()
