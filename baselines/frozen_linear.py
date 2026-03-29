"""
Baseline A: Frozen encoder + retrained linear head.

The encoder (first two layers) is frozen after initial training.
Only the final linear output head is updated online.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


class FrozenLinearBaseline(nn.Module):
    def __init__(self, n_classes: int, input_dim: int = 63):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128),       nn.ReLU(),
        )
        self.head = nn.Linear(128, n_classes)
        self.n_classes = n_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            h = self.encoder(x)
        return self.head(h)

    def add_class(self) -> int:
        old = self.head
        new_head = nn.Linear(old.in_features, self.n_classes + 1,
                              device=old.weight.device)
        with torch.no_grad():
            new_head.weight[:self.n_classes] = old.weight
            new_head.bias[:self.n_classes] = old.bias
        self.head = new_head
        self.n_classes += 1
        return self.n_classes - 1


class FrozenLinearTrainer:
    def __init__(self, model: FrozenLinearBaseline, device: torch.device, lr: float = 0.001):
        self.model = model
        self.device = device
        self.known_classes: set[int] = set()
        self.optimizer = torch.optim.Adam(model.head.parameters(), lr=lr)

    def step(self, features: torch.Tensor, labels: torch.Tensor) -> dict:
        features = features.to(self.device)
        labels = labels.to(self.device)

        for label in labels.tolist():
            if label not in self.known_classes:
                if label >= self.model.n_classes:
                    self.model.add_class()
                    self.optimizer = torch.optim.Adam(
                        self.model.head.parameters(), lr=0.001
                    )
                self.known_classes.add(label)

        self.model.train()
        self.optimizer.zero_grad()
        logits = self.model(features)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            self.model.eval()
            pred = self.model(features).argmax(1)
            acc = (pred == labels).float().mean().item()

        return {"loss": loss.item(), "acc": acc}
