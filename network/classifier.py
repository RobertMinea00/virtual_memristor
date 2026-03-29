"""
Full memristor-based hand-sign classifier.

Architecture:
  Input (63) -> BN -> MemristorLinear(256) -> ReLU
             -> MemristorLinear(128) -> ReLU
             -> MemristorLinear(N_classes)

The BN + first projection use standard nn.Linear so input normalisation
is stable before entering the analog layers.
"""

import torch
import torch.nn as nn
import yaml

from memristor.device_model import MemristorDeviceModel
from network.memristor_linear import MemristorLinear


class MemristorClassifier(nn.Module):
    def __init__(
        self,
        n_classes: int,
        input_dim: int = 63,
        hidden_dims: list[int] | None = None,
        device_model: MemristorDeviceModel | None = None,
        apply_noise: bool = True,
    ):
        super().__init__()

        if hidden_dims is None:
            with open("config/network.yaml") as f:
                cfg = yaml.safe_load(f)
            hidden_dims = cfg["hidden_dims"]

        if device_model is None:
            with open("config/memristor.yaml") as f:
                mem_cfg = yaml.safe_load(f)
            device_model = MemristorDeviceModel(mem_cfg)

        self.dm = device_model
        self.n_classes = n_classes
        self.apply_noise = apply_noise

        # Input normalisation — LayerNorm works with batch_size=1 (unlike BatchNorm1d)
        self.input_norm = nn.LayerNorm(input_dim)

        # Memristor hidden layers
        dims = [input_dim] + hidden_dims
        layers = []
        for i in range(len(dims) - 1):
            layers.append(MemristorLinear(
                dims[i], dims[i + 1],
                device_model=device_model,
                apply_noise_in_forward=apply_noise,
            ))
            layers.append(nn.ReLU())
        self.hidden = nn.Sequential(*layers)

        # Output layer — memristor, expandable
        self.output_layer = MemristorLinear(
            hidden_dims[-1], n_classes,
            device_model=device_model,
            apply_noise_in_forward=apply_noise,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        x = self.hidden(x)
        return self.output_layer(x)

    def memristor_layers(self) -> list[MemristorLinear]:
        """Return all MemristorLinear layers for bulk operations."""
        layers = [m for m in self.hidden.modules() if isinstance(m, MemristorLinear)]
        layers.append(self.output_layer)
        return layers

    def encode_all_shadows(self) -> None:
        """Sync all shadow weights -> conductances after an optimiser step."""
        for layer in self.memristor_layers():
            layer.encode_shadow()

    def add_class(self) -> int:
        """
        Expand the output layer by one neuron.
        Returns the index of the new class.
        """
        self.output_layer.expand_output(1)
        self.n_classes += 1
        return self.n_classes - 1

    def conductance_report(self) -> dict:
        report = {}
        for i, layer in enumerate(self.memristor_layers()):
            report[f"layer_{i}"] = layer.conductance_stats()
        return report
