"""
Drop-in replacement for nn.Linear using analog memristor crossbar simulation.

Shadow-parameter pattern:
  - W_shadow  (nn.Parameter, float32) — what the optimiser/gradient sees
  - G_pos, G_neg (buffers, float32) — what inference actually uses
  - After each update step, call encode_shadow() to sync shadow -> conductances

Forward pass uses G_pos/G_neg (with noise) for inference.
Backward pass flows through W_shadow via autograd (no straight-through needed
for the full layer — the decoder is a linear op so gradients are exact).
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from memristor.device_model import MemristorDeviceModel
from memristor.weight_mapper import WeightMapper
from memristor.crossbar import AnalogCrossbar


def _load_cfg():
    with open("config/memristor.yaml") as f:
        return yaml.safe_load(f)


class MemristorLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device_model: MemristorDeviceModel | None = None,
        apply_noise_in_forward: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.apply_noise = apply_noise_in_forward

        # Load device config if not provided
        if device_model is None:
            cfg = _load_cfg()
            device_model = MemristorDeviceModel(cfg)
        self.dm = device_model
        cfg = _load_cfg()

        self.mapper = WeightMapper(self.dm, w_max=1.0)
        self.crossbar = AnalogCrossbar(
            self.dm,
            tile_size=cfg["crossbar"]["tile_size"],
            adc_bits=cfg["crossbar"]["adc_bits"],
            dac_bits=cfg["crossbar"]["dac_bits"],
        )

        # Shadow weights — the optimiser updates these
        self.W_shadow = nn.Parameter(
            torch.empty(out_features, in_features)
        )
        nn.init.kaiming_uniform_(self.W_shadow, a=math.sqrt(5))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Conductance buffers — NOT parameters, updated manually
        G_pos_init, G_neg_init = self.mapper.encode(self.W_shadow.data)
        self.register_buffer("G_pos", G_pos_init.clone())
        self.register_buffer("G_neg", G_neg_init.clone())

        # Conductance at last write (for drift tracking)
        self.register_buffer("G_pos_at_write", G_pos_init.clone())
        self.register_buffer("G_neg_at_write", G_neg_init.clone())

        # Device-to-device variability (fixed at init, one offset per device)
        self.register_buffer("d2d_pos", self.dm.init_d2d_offset(
            (out_features, in_features), torch.device("cpu")))
        self.register_buffer("d2d_neg", self.dm.init_d2d_offset(
            (out_features, in_features), torch.device("cpu")))

        # Drift bookkeeping
        self._t_last_write = time.monotonic()
        self._forward_count = 0

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._forward_count += 1

        # Apply drift periodically
        if self._forward_count % self.dm.drift_every_n == 0:
            elapsed = time.monotonic() - self._t_last_write
            self.G_pos.copy_(
                self.dm.apply_drift(self.G_pos, self.G_pos_at_write, elapsed)
            )
            self.G_neg.copy_(
                self.dm.apply_drift(self.G_neg, self.G_neg_at_write, elapsed)
            )

        # Crossbar forward (includes read noise + d2d)
        out = self.crossbar.forward(
            x, self.G_pos, self.G_neg,
            self.d2d_pos, self.d2d_neg,
            self.mapper,
            apply_noise=self.apply_noise,
        )

        if self.bias is not None:
            out = out + self.bias

        return out

    # ------------------------------------------------------------------
    # Weight update — call after loss.backward()
    # ------------------------------------------------------------------

    def encode_shadow(self) -> None:
        """
        Project updated W_shadow back into conductance space.
        Call this after the optimiser step instead of the optimiser
        directly updating G_pos/G_neg (which are not parameters).
        """
        with torch.no_grad():
            self.mapper.update_w_max(self.W_shadow)
            G_pos_new, G_neg_new = self.mapper.encode(self.W_shadow)
            self.G_pos.copy_(G_pos_new)
            self.G_neg.copy_(G_neg_new)
            self.G_pos_at_write.copy_(G_pos_new)
            self.G_neg_at_write.copy_(G_neg_new)
            self._t_last_write = time.monotonic()

    # ------------------------------------------------------------------
    # Live class expansion (output dimension grows)
    # ------------------------------------------------------------------

    def expand_output(self, n_new: int) -> None:
        """Add n_new output neurons initialised to zero effective weight."""
        dev = self.W_shadow.device
        g_mid = (self.dm.g_max + self.dm.g_min) / 2.0

        # Expand shadow weight
        W_new_rows = torch.zeros(n_new, self.in_features, device=dev)
        new_W = nn.Parameter(torch.cat([self.W_shadow.data, W_new_rows], dim=0))
        self.W_shadow = new_W
        self.out_features += n_new

        # Expand conductance buffers
        G_mid_rows = torch.full((n_new, self.in_features), g_mid, device=dev)
        for buf_name in ("G_pos", "G_neg", "G_pos_at_write", "G_neg_at_write"):
            old = getattr(self, buf_name)
            new_buf = torch.cat([old, G_mid_rows.clone()], dim=0)
            self.register_buffer(buf_name, new_buf)

        # Expand d2d buffers
        for buf_name in ("d2d_pos", "d2d_neg"):
            old = getattr(self, buf_name)
            new_d2d = self.dm.init_d2d_offset((n_new, self.in_features), dev)
            self.register_buffer(buf_name, torch.cat([old, new_d2d], dim=0))

        # Expand bias
        if self.bias is not None:
            b_new = nn.Parameter(torch.cat([
                self.bias.data, torch.zeros(n_new, device=dev)
            ]))
            self.bias = b_new

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def conductance_stats(self) -> dict:
        g_eff = self.G_pos - self.G_neg
        return {
            "g_eff_mean": g_eff.mean().item(),
            "g_eff_std": g_eff.std().item(),
            "g_pos_mean": self.G_pos.mean().item(),
            "g_neg_mean": self.G_neg.mean().item(),
            "fraction_saturated": (
                (self.G_pos > 0.95 * self.dm.g_max).float().mean().item()
                + (self.G_neg > 0.95 * self.dm.g_max).float().mean().item()
            ) / 2,
        }
