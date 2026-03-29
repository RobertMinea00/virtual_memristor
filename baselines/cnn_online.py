"""
Baseline C: Small CNN treating the 21x3 landmark array as a spatial structure,
updated online with replay (same buffer size as memristor system for fairness).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class CNNOnlineBaseline(nn.Module):
    """
    Treats (21, 3) landmarks as a 1D spatial sequence:
      Conv1d over the 21 keypoints, then FC.
    """

    def __init__(self, n_classes: int, input_dim: int = 63):
        super().__init__()
        # Reshape: (B, 63) -> (B, 3, 21)  [3 channels: x, y, z; 21 spatial positions]
        self.conv = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 4, 128), nn.ReLU(),
            nn.Linear(128, n_classes),
        )
        self.n_classes = n_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        x = x.view(b, 21, 3).permute(0, 2, 1)  # (B, 3, 21)
        x = self.conv(x)
        x = x.flatten(1)
        return self.fc(x)

    def add_class(self) -> int:
        old = self.fc[-1]
        new_layer = nn.Linear(old.in_features, self.n_classes + 1,
                               device=old.weight.device)
        with torch.no_grad():
            new_layer.weight[:self.n_classes] = old.weight
            new_layer.bias[:self.n_classes] = old.bias
        self.fc[-1] = new_layer
        self.n_classes += 1
        return self.n_classes - 1


class CNNOnlineTrainer:
    def __init__(
        self,
        model: CNNOnlineBaseline,
        device: torch.device,
        lr: float = 0.001,
        buffer_size: int = 500,
    ):
        self.model = model
        self.device = device
        self.known_classes: set[int] = set()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self._buffer: list[tuple[torch.Tensor, int]] = []
        self._buffer_size = buffer_size
        self._seen = 0

    def _buffer_add(self, feat: torch.Tensor, label: int) -> None:
        self._seen += 1
        if len(self._buffer) < self._buffer_size:
            self._buffer.append((feat.cpu(), label))
        else:
            idx = random.randint(0, self._seen - 1)
            if idx < self._buffer_size:
                self._buffer[idx] = (feat.cpu(), label)

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
        loss = F.cross_entropy(self.model(features), labels)

        # Replay
        if len(self._buffer) >= 8:
            batch = random.sample(self._buffer, 8)
            rep_f = torch.stack([b[0] for b in batch]).to(self.device)
            rep_l = torch.tensor([b[1] for b in batch], device=self.device)
            loss = loss + F.cross_entropy(self.model(rep_f), rep_l)

        loss.backward()
        self.optimizer.step()

        for i in range(len(labels)):
            self._buffer_add(features[i].cpu(), labels[i].item())

        with torch.no_grad():
            self.model.eval()
            pred = self.model(features).argmax(1)
            acc = (pred == labels).float().mean().item()

        return {"loss": loss.item(), "acc": acc}
