"""
Metrics tracker for the continual learning benchmark.

Tracks per-system:
  - Adaptation speed: samples-to-80%-accuracy on each new class
  - Backward transfer (BWT): accuracy drop on old classes after learning new ones
  - Inference latency (torch.cuda.Event)
  - Update latency
  - GPU memory usage
  - Conductance distribution (memristor system only)
"""

import time
from collections import defaultdict
import torch


class MetricsTracker:
    def __init__(self, system_name: str, device: torch.device):
        self.name = system_name
        self.device = device
        self.log_interval = 10

        # Per-step log
        self.step_log: list[dict] = []

        # Adaptation speed: class -> list of (sample_idx, acc)
        self._acc_by_class: dict[int, list[tuple[int, float]]] = defaultdict(list)
        self._adaptation_speed: dict[int, int | None] = {}  # class -> n_samples or None

        # BWT: snapshot of per-class acc just before adding a new class
        self._acc_snapshot_before: dict[int, dict[int, float]] = {}
        self._acc_snapshot_after: dict[int, dict[int, float]] = {}

        # Latency
        self._latency_samples: list[float] = []
        self._update_latency_samples: list[float] = []

        self._sample_count = 0

    # ------------------------------------------------------------------
    # Log one training step
    # ------------------------------------------------------------------

    def record_step(
        self,
        result: dict,
        per_class_acc: dict[int, float] | None = None,
        latency_ms: float | None = None,
        update_latency_ms: float | None = None,
        conductance_report: dict | None = None,
    ) -> None:
        self._sample_count += 1

        entry = {
            "sample": self._sample_count,
            **result,
        }
        if per_class_acc:
            entry["per_class_acc"] = per_class_acc
        if latency_ms is not None:
            entry["latency_ms"] = latency_ms
            self._latency_samples.append(latency_ms)
        if update_latency_ms is not None:
            entry["update_latency_ms"] = update_latency_ms
            self._update_latency_samples.append(update_latency_ms)
        if conductance_report:
            entry["conductance"] = conductance_report

        self.step_log.append(entry)

        # Track adaptation speed
        if per_class_acc:
            for cls, acc in per_class_acc.items():
                self._acc_by_class[cls].append((self._sample_count, acc))
                if cls not in self._adaptation_speed and acc >= 0.80:
                    # Find first sample where class was introduced
                    self._adaptation_speed[cls] = self._sample_count

    def snapshot_before_new_class(self, per_class_acc: dict[int, float], new_class: int) -> None:
        self._acc_snapshot_before[new_class] = dict(per_class_acc)

    def snapshot_after_new_class(self, per_class_acc: dict[int, float], new_class: int) -> None:
        self._acc_snapshot_after[new_class] = dict(per_class_acc)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        bwt = self._compute_bwt()
        return {
            "system": self.name,
            "total_samples": self._sample_count,
            "adaptation_speed": dict(self._adaptation_speed),
            "bwt": bwt,
            "mean_inference_latency_ms": (
                sum(self._latency_samples) / len(self._latency_samples)
                if self._latency_samples else None
            ),
            "mean_update_latency_ms": (
                sum(self._update_latency_samples) / len(self._update_latency_samples)
                if self._update_latency_samples else None
            ),
            "gpu_memory_mb": torch.cuda.memory_allocated(self.device) / 1e6
            if self.device.type == "cuda" else 0,
        }

    def _compute_bwt(self) -> float | None:
        """
        Average accuracy drop on old classes after learning new ones.
        BWT = mean over tasks T: acc(T, after learning T+1) - acc(T, just before T+1)
        Negative BWT = forgetting.
        """
        diffs = []
        for new_cls in self._acc_snapshot_before:
            before = self._acc_snapshot_before[new_cls]
            after = self._acc_snapshot_after.get(new_cls, {})
            for cls in before:
                if cls in after and cls != new_cls:
                    diffs.append(after[cls] - before[cls])
        return sum(diffs) / len(diffs) if diffs else None

    def print_summary(self) -> None:
        s = self.summary()
        print(f"\n{'='*50}")
        print(f"System: {s['system']}")
        print(f"  Total samples processed: {s['total_samples']}")
        print(f"  Adaptation speed (samples to 80%): {s['adaptation_speed']}")
        print(f"  BWT (forgetting): {s['bwt']:.4f}" if s['bwt'] is not None else "  BWT: N/A")
        if s["mean_inference_latency_ms"]:
            print(f"  Inference latency: {s['mean_inference_latency_ms']:.2f} ms")
        if s["mean_update_latency_ms"]:
            print(f"  Update latency:    {s['mean_update_latency_ms']:.2f} ms")
        print(f"  GPU memory: {s['gpu_memory_mb']:.1f} MB")
        print(f"{'='*50}\n")
