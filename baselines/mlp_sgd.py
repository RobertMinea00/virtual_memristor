"""
Baseline B: Small MLP updated online with Adam, no replay, no EWC.
Demonstrates catastrophic forgetting clearly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPOnlineBaseline(nn.Module):
    def __init__(self, n_classes: int, input_dim: int = 63):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128),       nn.ReLU(),
            nn.Linear(128, n_classes),
        )
        self.n_classes = n_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def add_class(self) -> int:
        old = self.net[-1]
        new_layer = nn.Linear(old.in_features, self.n_classes + 1,
                               device=old.weight.device)
        with torch.no_grad():
            new_layer.weight[:self.n_classes] = old.weight
            new_layer.bias[:self.n_classes] = old.bias
        self.net[-1] = new_layer
        self.n_classes += 1
        return self.n_classes - 1


class MLPOnlineTrainer:
    def __init__(self, model: MLPOnlineBaseline, device: torch.device, lr: float = 0.001):
        self.model = model
        self.device = device
        self.known_classes: set[int] = set()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def step(self, features: torch.Tensor, labels: torch.Tensor) -> dict:
        features = features.to(self.device)
        labels = labels.to(self.device)

        for label in labels.tolist():
            if label not in self.known_classes:
                if label >= self.model.n_classes:
                    self.model.add_class()
                    self.optimizer = torch.optim.Adam(
                        self.model.parameters(), lr=0.001
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
