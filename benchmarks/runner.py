"""
Benchmark runner.

Instantiates all four systems (memristor + 3 baselines), feeds them the
identical synthetic data stream, logs all metrics side by side.

Usage:
    python scripts/run_benchmark.py
"""

import time
import torch
import yaml

from memristor.device_model import MemristorDeviceModel
from network.classifier import MemristorClassifier
from learning.continual_trainer import ContinualTrainer
from baselines.frozen_linear import FrozenLinearBaseline, FrozenLinearTrainer
from baselines.mlp_sgd import MLPOnlineBaseline, MLPOnlineTrainer
from baselines.cnn_online import CNNOnlineBaseline, CNNOnlineTrainer
from benchmarks.data_stream import HandSignStream
from benchmarks.metrics import MetricsTracker


def _make_start_event():
    e = torch.cuda.Event(enable_timing=True)
    e.record()
    return e


def _elapsed_ms(start_event, end_event) -> float:
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event)


class BenchmarkRunner:
    def __init__(self, cfg_path: str = "config"):
        with open(f"{cfg_path}/network.yaml") as f:
            net_cfg = yaml.safe_load(f)
        with open(f"{cfg_path}/benchmark.yaml") as f:
            bench_cfg = yaml.safe_load(f)
        with open(f"{cfg_path}/memristor.yaml") as f:
            mem_cfg = yaml.safe_load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_classes = net_cfg["initial_classes"]
        seed = bench_cfg["seed"]
        torch.manual_seed(seed)

        # ---- Memristor system ----
        dm = MemristorDeviceModel(mem_cfg)
        self.mem_model = MemristorClassifier(
            n_classes=n_classes,
            input_dim=net_cfg["input_dim"],
            hidden_dims=net_cfg["hidden_dims"],
            device_model=dm,
        ).to(self.device)
        self.mem_trainer = ContinualTrainer(self.mem_model, self.device)

        # ---- Baseline A: Frozen Linear ----
        self.frozen_model = FrozenLinearBaseline(n_classes).to(self.device)
        self.frozen_trainer = FrozenLinearTrainer(self.frozen_model, self.device)

        # ---- Baseline B: MLP SGD ----
        self.mlp_model = MLPOnlineBaseline(n_classes).to(self.device)
        self.mlp_trainer = MLPOnlineTrainer(self.mlp_model, self.device)

        # ---- Baseline C: CNN Online ----
        self.cnn_model = CNNOnlineBaseline(n_classes).to(self.device)
        self.cnn_trainer = CNNOnlineTrainer(self.cnn_model, self.device)

        # ---- Metrics ----
        self.trackers = {
            "memristor":    MetricsTracker("memristor",    self.device),
            "frozen_linear": MetricsTracker("frozen_linear", self.device),
            "mlp_sgd":      MetricsTracker("mlp_sgd",      self.device),
            "cnn_online":   MetricsTracker("cnn_online",   self.device),
        }

        # ---- Data stream ----
        self.stream = HandSignStream(
            n_initial_classes=n_classes,
            samples_per_class=bench_cfg["samples_per_class_before_new"],
            device=self.device,
            seed=seed,
        )

    def run(self) -> dict:
        stream = self.stream.generate_stream()
        print(f"Running benchmark on {len(stream)} samples, device={self.device}")

        systems = [
            ("memristor",    self.mem_trainer,    self.mem_model),
            ("frozen_linear", self.frozen_trainer, self.frozen_model),
            ("mlp_sgd",      self.mlp_trainer,    self.mlp_model),
            ("cnn_online",   self.cnn_trainer,    self.cnn_model),
        ]

        for sample_idx, (feats, labels) in enumerate(stream):
            feats = feats.to(self.device)
            labels = labels.to(self.device)

            for sys_name, trainer, model in systems:
                tracker = self.trackers[sys_name]

                # Time the update
                t0 = time.perf_counter()
                result = trainer.step(feats, labels)
                update_ms = (time.perf_counter() - t0) * 1000

                # Time inference only
                model.eval()
                if self.device.type == "cuda":
                    start_e = torch.cuda.Event(enable_timing=True)
                    end_e = torch.cuda.Event(enable_timing=True)
                    start_e.record()
                    with torch.no_grad():
                        _ = model(feats)
                    end_e.record()
                    torch.cuda.synchronize()
                    inf_ms = start_e.elapsed_time(end_e)
                else:
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        _ = model(feats)
                    inf_ms = (time.perf_counter() - t0) * 1000

                # Conductance report (memristor only)
                cond = None
                if sys_name == "memristor" and hasattr(model, "conductance_report"):
                    cond = model.conductance_report()

                tracker.record_step(
                    result,
                    latency_ms=inf_ms,
                    update_latency_ms=update_ms,
                    conductance_report=cond,
                )

            if (sample_idx + 1) % 50 == 0:
                print(f"  Step {sample_idx+1}/{len(stream)}")

        print("\n--- RESULTS ---")
        summaries = {}
        for name, tracker in self.trackers.items():
            tracker.print_summary()
            summaries[name] = tracker.summary()

        return summaries
