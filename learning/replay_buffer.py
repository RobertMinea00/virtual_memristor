"""
GPU-resident episodic replay buffer using reservoir sampling (Vitter's Algorithm R).

The buffer stores (feature, label) pairs. All tensors live on GPU as a
pre-allocated block to avoid host-device transfer during replay.
"""

import torch
import random


class ReplayBuffer:
    def __init__(self, capacity: int, feature_dim: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.size = 0
        self.total_seen = 0

        # Pre-allocate on GPU
        self.features = torch.zeros(capacity, feature_dim, device=device)
        self.labels = torch.zeros(capacity, dtype=torch.long, device=device)

    def add(self, feature: torch.Tensor, label: int) -> None:
        """Add one sample using reservoir sampling."""
        self.total_seen += 1
        if self.size < self.capacity:
            idx = self.size
            self.size += 1
        else:
            idx = random.randint(0, self.total_seen - 1)
            if idx >= self.capacity:
                return  # sample discarded — reservoir full and not selected

        self.features[idx] = feature.detach().to(self.device)
        self.labels[idx] = label

    def add_batch(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        for i in range(len(labels)):
            self.add(features[i], labels[i].item())

    def sample(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample n items uniformly. Returns (features, labels) on device."""
        n = min(n, self.size)
        indices = torch.randint(0, self.size, (n,), device=self.device)
        return self.features[indices], self.labels[indices]

    def __len__(self) -> int:
        return self.size
