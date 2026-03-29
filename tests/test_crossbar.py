"""
Tests for crossbar and weight mapper.

Critical regression: with ideal device (no noise, infinite resolution,
no quantisation), the crossbar should produce output identical to
a standard nn.Linear forward pass.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from memristor.device_model import MemristorDeviceModel
from memristor.weight_mapper import WeightMapper
from memristor.crossbar import AnalogCrossbar

IDEAL_CFG = {
    "g_min": 1e-6,
    "g_max": 1e-4,
    "n_levels": 0,
    "noise": {"sigma_write": 0.0, "sigma_read": 0.0, "sigma_d2d": 0.0},
    "drift": {"nu": 0.0, "t0": 1.0, "update_every_n": 30},
}


def make_ideal_system(in_f=32, out_f=16):
    dm = MemristorDeviceModel(IDEAL_CFG)
    mapper = WeightMapper(dm, w_max=1.0)
    crossbar = AnalogCrossbar(dm, tile_size=128, adc_bits=0, dac_bits=0)  # no quantisation
    W = torch.randn(out_f, in_f) * 0.5
    # Calibrate w_max so no weight is clipped during encode
    mapper.update_w_max(W)
    G_pos, G_neg = mapper.encode(W)
    d2d = torch.ones_like(G_pos)
    return dm, mapper, crossbar, W, G_pos, G_neg, d2d


def test_ideal_crossbar_matches_linear():
    """Ideal crossbar == nn.Linear (no noise, no ADC/DAC quantisation)."""
    dm, mapper, crossbar, W, G_pos, G_neg, d2d = make_ideal_system(32, 16)
    x = torch.randn(8, 32)
    expected = F.linear(x, W)
    actual = crossbar.forward(x, G_pos, G_neg, d2d, d2d, mapper, apply_noise=False)
    assert torch.allclose(expected, actual, atol=1e-4), \
        f"Max diff: {(expected - actual).abs().max().item():.2e}"


def test_differential_pair_preserves_sign():
    """Negative weights should survive differential-pair encoding/decoding."""
    dm, mapper, crossbar, W, G_pos, G_neg, d2d = make_ideal_system(8, 8)
    W_decoded = mapper.decode(G_pos, G_neg)
    assert torch.allclose(W, W_decoded, atol=1e-5)


def test_tiling_equivalent_to_no_tiling():
    """Small tile forces tiling; result should match large-tile (no tiling)."""
    dm = MemristorDeviceModel(IDEAL_CFG)
    mapper = WeightMapper(dm, w_max=1.0)
    W = torch.randn(64, 64) * 0.5
    G_pos, G_neg = mapper.encode(W)
    d2d = torch.ones_like(G_pos)
    x = torch.randn(4, 64)

    cb_tiled = AnalogCrossbar(dm, tile_size=16, adc_bits=0, dac_bits=0)
    cb_full  = AnalogCrossbar(dm, tile_size=256, adc_bits=0, dac_bits=0)

    out_tiled = cb_tiled.forward(x, G_pos, G_neg, d2d, d2d, mapper, apply_noise=False)
    out_full  = cb_full.forward(x, G_pos, G_neg, d2d, d2d, mapper, apply_noise=False)

    assert torch.allclose(out_tiled, out_full, atol=1e-4), \
        f"Tiling mismatch, max diff: {(out_tiled - out_full).abs().max():.2e}"


def test_adc_dac_quantisation_reduces_accuracy():
    """With 8-bit ADC/DAC, output should deviate slightly from ideal."""
    dm = MemristorDeviceModel(IDEAL_CFG)
    mapper = WeightMapper(dm, w_max=1.0)
    W = torch.randn(16, 32) * 0.5
    G_pos, G_neg = mapper.encode(W)
    d2d = torch.ones_like(G_pos)
    x = torch.randn(8, 32)

    cb_ideal = AnalogCrossbar(dm, tile_size=128, adc_bits=0, dac_bits=0)
    cb_quant = AnalogCrossbar(dm, tile_size=128, adc_bits=8, dac_bits=8)

    out_ideal = cb_ideal.forward(x, G_pos, G_neg, d2d, d2d, mapper, apply_noise=False)
    out_quant = cb_quant.forward(x, G_pos, G_neg, d2d, d2d, mapper, apply_noise=False)

    diff = (out_ideal - out_quant).abs().max().item()
    assert diff > 1e-7, "Quantisation should introduce some error"
    assert diff < 1.0,  "Quantisation error should be small for 8-bit"
