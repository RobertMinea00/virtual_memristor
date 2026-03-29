"""
Analog crossbar array abstraction.

Handles:
  - Tiling of large weight matrices into crossbar-sized tiles
  - DAC quantisation of inputs
  - ADC quantisation of outputs
  - Differential-pair forward pass with read noise + drift applied

The crossbar itself is stateless; all buffers (G_pos, G_neg, d2d, etc.)
live in MemristorLinear and are passed in.
"""

import torch
import torch.nn.functional as F
from .device_model import MemristorDeviceModel
from .weight_mapper import WeightMapper


def _quantise_to_bits(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Uniform quantisation of x to `bits` bits over its dynamic range."""
    if bits <= 0:
        return x
    levels = 2 ** bits - 1
    x_min = x.min()
    x_max = x.max()
    span = (x_max - x_min).clamp(min=1e-9)
    x_norm = (x - x_min) / span                       # [0, 1]
    x_q = (x_norm * levels).round() / levels           # quantised [0, 1]
    return x_q * span + x_min


class AnalogCrossbar:
    """
    Performs a matrix-vector product through a simulated crossbar.

    Args:
        device_model: MemristorDeviceModel instance
        tile_size:    maximum rows/cols per physical crossbar tile
        adc_bits:     ADC resolution (0 = infinite)
        dac_bits:     DAC resolution (0 = infinite)
    """

    def __init__(
        self,
        device_model: MemristorDeviceModel,
        tile_size: int = 128,
        adc_bits: int = 8,
        dac_bits: int = 8,
    ):
        self.dm = device_model
        self.tile_size = tile_size
        self.adc_bits = adc_bits
        self.dac_bits = dac_bits

    def forward(
        self,
        x: torch.Tensor,
        G_pos: torch.Tensor,
        G_neg: torch.Tensor,
        d2d_pos: torch.Tensor,
        d2d_neg: torch.Tensor,
        mapper: WeightMapper,
        apply_noise: bool = True,
    ) -> torch.Tensor:
        """
        x      : (batch, in_features)
        G_pos  : (out_features, in_features)
        G_neg  : (out_features, in_features)
        Returns: (batch, out_features)
        """
        out_features, in_features = G_pos.shape

        # DAC quantisation of input voltages
        x_q = _quantise_to_bits(x, self.dac_bits)

        # Tile loop — accumulate output across tiles
        output = torch.zeros(x.shape[0], out_features, device=x.device, dtype=x.dtype)

        for row_start in range(0, out_features, self.tile_size):
            row_end = min(row_start + self.tile_size, out_features)
            for col_start in range(0, in_features, self.tile_size):
                col_end = min(col_start + self.tile_size, in_features)

                gp = G_pos[row_start:row_end, col_start:col_end]
                gn = G_neg[row_start:row_end, col_start:col_end]
                dp = d2d_pos[row_start:row_end, col_start:col_end]
                dn = d2d_neg[row_start:row_end, col_start:col_end]

                if apply_noise:
                    gp = self.dm.apply_read_noise(gp, dp)
                    gn = self.dm.apply_read_noise(gn, dn)

                W_eff = mapper.decode(gp, gn)

                tile_out = F.linear(
                    x_q[:, col_start:col_end], W_eff
                )  # (batch, tile_rows)

                # ADC quantisation of tile output
                tile_out = _quantise_to_bits(tile_out, self.adc_bits)

                output[:, row_start:row_end] += tile_out

        return output
