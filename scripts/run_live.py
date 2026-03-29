"""
Launch live webcam inference with the memristor classifier.

Usage:
    .venv/Scripts/python scripts/run_live.py --class-names A B C D E
    .venv/Scripts/python scripts/run_live.py --model-path checkpoints/model.pt
"""

import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml

from memristor.device_model import MemristorDeviceModel
from network.classifier import MemristorClassifier
from pipeline.inference_loop import InferenceLoop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--class-names", nargs="+", default=[f"cls{i}" for i in range(5)])
    parser.add_argument("--model-path", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open("config/network.yaml") as f:
        net_cfg = yaml.safe_load(f)
    with open("config/memristor.yaml") as f:
        mem_cfg = yaml.safe_load(f)

    dm = MemristorDeviceModel(mem_cfg)
    n_classes = max(net_cfg["initial_classes"], len(args.class_names))
    model = MemristorClassifier(
        n_classes=n_classes,
        input_dim=net_cfg["input_dim"],
        hidden_dims=net_cfg["hidden_dims"],
        device_model=dm,
    ).to(device)

    if args.model_path and Path(args.model_path).exists():
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Loaded model from {args.model_path}")
    else:
        print("No model checkpoint provided — using randomly initialised weights.")

    loop = InferenceLoop(
        model=model,
        device=device,
        class_names=args.class_names,
    )

    print("Press Q in the OpenCV window to quit.")
    loop.run(show_window=True)


if __name__ == "__main__":
    main()
