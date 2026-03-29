"""
Bidirectional mapping between float32 shadow weights and
differential-pair conductances (G_pos, G_neg).

Encoding:
  W in [-W_max, +W_max]
  G_pos = g_mid + W * scale / 2   (always in [g_min, g_max])
  G_neg = g_mid - W * scale / 2
  => W_eff = (G_pos - G_neg) / scale

where  g_mid = (g_max + g_min) / 2
       scale = (g_max - g_min) / (2 * W_max)
       W_max is determined per-layer at init as the max absolute weight value,
       or passed explicitly.
"""

import torch
from .device_model import MemristorDeviceModel


class WeightMapper:
    def __init__(self, device_model: MemristorDeviceModel, w_max: float = 1.0):
        self.dm = device_model
        self.w_max = w_max
        self.g_mid = (device_model.g_max + device_model.g_min) / 2.0
        # scale: maps W=w_max -> G=g_max, W=-w_max -> G=g_min (for G_pos column)
        self.scale = (device_model.g_max - device_model.g_min) / (2.0 * w_max)

    def encode(
        self, W_shadow: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Float weights -> (G_pos, G_neg) after quantisation and write noise.
        Returns programmed conductance pair tensors on the same device as W_shadow.
        """
        W_clipped = W_shadow.clamp(-self.w_max, self.w_max)
        G_pos_target = self.g_mid + W_clipped * self.scale
        G_neg_target = self.g_mid - W_clipped * self.scale
        G_pos = self.dm.program(G_pos_target)
        G_neg = self.dm.program(G_neg_target)
        return G_pos, G_neg

    def decode(
        self, G_pos: torch.Tensor, G_neg: torch.Tensor
    ) -> torch.Tensor:
        """
        (G_pos, G_neg) -> effective float weight matrix.

        Encoding: G_pos = g_mid + W * scale, G_neg = g_mid - W * scale
        => G_pos - G_neg = 2 * W * scale  =>  W = (G_pos - G_neg) / (2 * scale)
        """
        return (G_pos - G_neg) / (2.0 * self.scale)

    def update_w_max(self, W_shadow: torch.Tensor) -> None:
        """Re-calibrate the scale when weights grow beyond w_max."""
        new_max = W_shadow.abs().max().item()
        if new_max > self.w_max:
            self.w_max = new_max * 1.1  # 10% headroom
            self.scale = (self.dm.g_max - self.dm.g_min) / (2.0 * self.w_max)
