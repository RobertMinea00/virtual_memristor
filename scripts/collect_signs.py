"""
Interactive tool to record hand-sign samples for a new class.

Usage:
    .venv/Scripts/python scripts/collect_signs.py --class-name "ThumbsUp" --n-samples 100

The samples are saved to data/<class_name>.pt for offline benchmark use,
and optionally fed live into the continual trainer.
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
from learning.continual_trainer import ContinualTrainer
from pipeline.inference_loop import InferenceLoop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--class-name", required=True)
    parser.add_argument("--class-idx", type=int, default=None)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--no-train", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open("config/network.yaml") as f:
        net_cfg = yaml.safe_load(f)
    with open("config/memristor.yaml") as f:
        mem_cfg = yaml.safe_load(f)

    n_classes = net_cfg["initial_classes"]
    dm = MemristorDeviceModel(mem_cfg)
    model = MemristorClassifier(
        n_classes=n_classes,
        input_dim=net_cfg["input_dim"],
        hidden_dims=net_cfg["hidden_dims"],
        device_model=dm,
    ).to(device)

    if args.model_path and Path(args.model_path).exists():
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Loaded model from {args.model_path}")

    trainer = None if args.no_train else ContinualTrainer(model, device)

    class_idx = args.class_idx
    if class_idx is None:
        class_idx = model.n_classes  # append as next class

    loop = InferenceLoop(
        model=model,
        device=device,
        class_names=[f"cls{i}" for i in range(model.n_classes)] + [args.class_name],
        trainer=trainer,
    )

    print(f"\nRecording class '{args.class_name}' (index {class_idx})")
    print("Show your hand sign. Press Q to quit early.\n")

    collected = loop.run(
        label_for_training=class_idx,
        max_samples=args.n_samples,
        show_window=True,
    )

    if collected:
        save_dir = Path("data")
        save_dir.mkdir(exist_ok=True)
        save_path = save_dir / f"{args.class_name}.pt"
        torch.save({
            "features": torch.stack(collected),
            "label": class_idx,
            "class_name": args.class_name,
        }, save_path)
        print(f"Saved {len(collected)} samples to {save_path}")

    if args.model_path and not args.no_train:
        torch.save(model.state_dict(), args.model_path)
        print(f"Updated model saved to {args.model_path}")


if __name__ == "__main__":
    main()
