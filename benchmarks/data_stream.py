"""
Simulated hand-sign data stream for offline benchmarking.

Generates synthetic 63-element feature vectors per class.
Each class has a distinct mean and within-class variance to simulate
real landmark distributions. Can also replay from recorded .pt files.
"""

import torch
from pathlib import Path


class HandSignStream:
    def __init__(
        self,
        n_initial_classes: int = 5,
        samples_per_class: int = 50,
        feature_dim: int = 63,
        device: torch.device | None = None,
        seed: int = 42,
    ):
        self.device = device or torch.device("cpu")
        self.feature_dim = feature_dim
        self.n_initial_classes = n_initial_classes
        self.samples_per_class = samples_per_class
        self.seed = seed

        torch.manual_seed(seed)

        # Fixed class centroids — distinct random means in landmark space
        self._centroids: dict[int, torch.Tensor] = {}
        self._class_std: dict[int, float] = {}
        for c in range(n_initial_classes):
            self._centroids[c] = torch.randn(feature_dim) * 0.5
            self._class_std[c] = 0.05 + torch.rand(1).item() * 0.05

        self._all_classes = list(range(n_initial_classes))

    def add_class(self, class_idx: int | None = None) -> int:
        """Add a new class with a random centroid. Returns the class index."""
        if class_idx is None:
            class_idx = max(self._centroids.keys()) + 1
        if class_idx not in self._centroids:
            self._centroids[class_idx] = torch.randn(self.feature_dim) * 0.5
            self._class_std[class_idx] = 0.05 + torch.rand(1).item() * 0.05
            self._all_classes.append(class_idx)
        return class_idx

    def sample_class(
        self,
        class_idx: int,
        n: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (features, labels) for n samples of a given class."""
        mean = self._centroids[class_idx]
        std = self._class_std[class_idx]
        feats = mean.unsqueeze(0) + torch.randn(n, self.feature_dim) * std
        labels = torch.full((n,), class_idx, dtype=torch.long)
        return feats.to(self.device), labels.to(self.device)

    def generate_stream(
        self,
        schedule: list[dict] | None = None,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate a full benchmark stream.

        schedule is a list of dicts like:
          [{"class": 0, "n": 50}, {"class": 1, "n": 50}, ..., {"new_class": True, "n": 50}]

        If None, uses the default: cycle through initial classes, then add one more.
        Returns list of (feature, label) 1-sample pairs.
        """
        if schedule is None:
            schedule = []
            for c in range(self.n_initial_classes):
                schedule.append({"class": c, "n": self.samples_per_class})
            new_cls = self.add_class()
            schedule.append({"class": new_cls, "n": self.samples_per_class})
            # One more to show forgetting
            for c in range(self.n_initial_classes):
                schedule.append({"class": c, "n": 20})

        stream = []
        for entry in schedule:
            if "new_class" in entry:
                cls = self.add_class()
            else:
                cls = entry["class"]
                if cls not in self._centroids:
                    self.add_class(cls)
            n = entry.get("n", self.samples_per_class)
            feats, labels = self.sample_class(cls, n)
            for i in range(n):
                stream.append((feats[i:i+1], labels[i:i+1]))

        return stream

    # ------------------------------------------------------------------
    # Save / load recorded real-world streams
    # ------------------------------------------------------------------

    def save(self, path: str, stream: list) -> None:
        feats = torch.cat([s[0] for s in stream])
        labels = torch.cat([s[1] for s in stream])
        torch.save({"features": feats, "labels": labels}, path)

    @staticmethod
    def load(path: str, device: torch.device | None = None):
        data = torch.load(path, map_location=device or "cpu")
        feats = data["features"]
        labels = data["labels"]
        return [(feats[i:i+1], labels[i:i+1]) for i in range(len(labels))]
