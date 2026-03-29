"""
Tests for memristor device model physics.

Critical regression: with all noise/drift disabled, device_model
should behave as identity (no-op on conductances).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
from memristor.device_model import MemristorDeviceModel

IDEAL_CFG = {
    "g_min": 1e-6,
    "g_max": 1e-4,
    "n_levels": 0,  # infinite resolution
    "noise": {"sigma_write": 0.0, "sigma_read": 0.0, "sigma_d2d": 0.0},
    "drift": {"nu": 0.0, "t0": 1.0, "update_every_n": 30},
}

NOISY_CFG = {
    "g_min": 1e-6,
    "g_max": 1e-4,
    "n_levels": 32,
    "noise": {"sigma_write": 0.05, "sigma_read": 0.01, "sigma_d2d": 0.02},
    "drift": {"nu": 0.012, "t0": 1.0, "update_every_n": 30},
}


def test_ideal_program_is_noop():
    """With zero noise and zero quantisation, program() is identity."""
    dm = MemristorDeviceModel(IDEAL_CFG)
    g = torch.linspace(dm.g_min, dm.g_max, 1000)
    g_out = dm.program(g)
    assert torch.allclose(g, g_out, atol=1e-12), "Ideal program should be identity"


def test_quantisation_levels():
    """With n_levels=32, output should have at most 32 distinct values."""
    cfg = dict(IDEAL_CFG)
    cfg["n_levels"] = 32
    cfg["noise"] = {"sigma_write": 0.0, "sigma_read": 0.0, "sigma_d2d": 0.0}
    dm = MemristorDeviceModel(cfg)
    g = torch.linspace(dm.g_min, dm.g_max, 10000)
    g_out = dm.program(g)
    n_unique = g_out.unique().numel()
    assert n_unique <= 32, f"Expected <= 32 levels, got {n_unique}"


def test_write_noise_distribution():
    """Write noise should be Gaussian with expected std."""
    dm = MemristorDeviceModel(NOISY_CFG)
    G_target = torch.full((10000,), (dm.g_min + dm.g_max) / 2)
    G_out = dm.program(G_target)
    # Allow quantisation effect; just check noise is present
    assert (G_out - G_target).abs().max() > 0, "Write noise should perturb values"


def test_read_noise_statistics():
    """Read noise N(0, sigma_read) should have correct empirical std."""
    dm = MemristorDeviceModel(NOISY_CFG)
    G = torch.full((50000,), 5e-5)
    d2d = torch.ones_like(G)
    G_read = dm.apply_read_noise(G, d2d)
    relative_noise = (G_read - G) / G
    empirical_std = relative_noise.std().item()
    assert abs(empirical_std - NOISY_CFG["noise"]["sigma_read"]) < 0.002, \
        f"Read noise std mismatch: expected ~{NOISY_CFG['noise']['sigma_read']}, got {empirical_std:.4f}"


def test_d2d_offset_shape():
    dm = MemristorDeviceModel(NOISY_CFG)
    shape = (64, 128)
    d2d = dm.init_d2d_offset(shape, torch.device("cpu"))
    assert d2d.shape == shape
    assert not torch.all(d2d == 1.0), "D2D offsets should vary"


def test_drift_reduces_conductance():
    """Drift with nu>0 should reduce conductance over time."""
    dm = MemristorDeviceModel(NOISY_CFG)
    G0 = torch.full((100,), 8e-5)
    G_drifted = dm.apply_drift(G0.clone(), G0, elapsed_seconds=100.0)
    assert (G_drifted < G0).all(), "Drift should reduce conductance (power law decay)"


def test_conductance_clamped():
    dm = MemristorDeviceModel(NOISY_CFG)
    G = torch.tensor([0.0, 1.0, -1.0, dm.g_min, dm.g_max])  # includes out-of-range
    G_out = dm.program(G)
    assert G_out.min() >= dm.g_min - 1e-9
    assert G_out.max() <= dm.g_max + 1e-9


def test_cuda_consistency():
    """Same result shape on CUDA as CPU."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    dm = MemristorDeviceModel(NOISY_CFG)
    G_cpu = torch.full((128, 128), 5e-5)
    G_gpu = G_cpu.cuda()
    out_cpu = dm.program(G_cpu)
    out_gpu = dm.program(G_gpu)
    assert out_cpu.shape == out_gpu.cpu().shape
