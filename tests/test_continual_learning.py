"""
Tests for continual learning components.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest
from learning.replay_buffer import ReplayBuffer
from memristor.device_model import MemristorDeviceModel
from network.classifier import MemristorClassifier
from learning.continual_trainer import ContinualTrainer

IDEAL_CFG = {
    "g_min": 1e-6, "g_max": 1e-4, "n_levels": 0,
    "noise": {"sigma_write": 0.0, "sigma_read": 0.0, "sigma_d2d": 0.0},
    "drift": {"nu": 0.0, "t0": 1.0, "update_every_n": 30},
}


def make_model(n_classes=3):
    dm = MemristorDeviceModel(IDEAL_CFG)
    return MemristorClassifier(
        n_classes=n_classes,
        input_dim=63,
        hidden_dims=[64, 32],
        device_model=dm,
        apply_noise=False,
    )


# ---- Replay buffer tests ----

def test_replay_buffer_fills():
    buf = ReplayBuffer(capacity=100, feature_dim=63, device=torch.device("cpu"))
    for i in range(150):
        buf.add(torch.randn(63), i % 5)
    assert len(buf) == 100  # capped at capacity


def test_replay_buffer_sample_size():
    buf = ReplayBuffer(capacity=50, feature_dim=63, device=torch.device("cpu"))
    for i in range(30):
        buf.add(torch.randn(63), 0)
    feats, labels = buf.sample(10)
    assert feats.shape == (10, 63)
    assert labels.shape == (10,)


def test_replay_buffer_partial_fill():
    buf = ReplayBuffer(capacity=100, feature_dim=63, device=torch.device("cpu"))
    for i in range(20):
        buf.add(torch.randn(63), 0)
    feats, labels = buf.sample(50)  # ask for more than available
    assert feats.shape == (20, 63)


# ---- Class expansion tests ----

def test_add_class_does_not_reset_existing():
    """Adding a new class should not alter existing output neurons."""
    model = make_model(n_classes=3)
    W_before = model.output_layer.G_pos[:3].clone()
    model.add_class()
    W_after = model.output_layer.G_pos[:3]
    assert torch.allclose(W_before, W_after), "Existing neurons should be unchanged"


def test_add_class_increments_n_classes():
    model = make_model(n_classes=3)
    assert model.n_classes == 3
    model.add_class()
    assert model.n_classes == 4


def test_add_class_new_neuron_near_zero():
    """New output neuron should have near-zero effective weight (g_mid differential)."""
    model = make_model(n_classes=3)
    model.add_class()
    # Last row of effective weight should be near zero
    W_eff = model.output_layer.G_pos[-1] - model.output_layer.G_neg[-1]
    # Both G_pos and G_neg initialised to g_mid, so W_eff should be 0
    assert W_eff.abs().max() < 1e-9, "New neuron should have zero effective weight"


# ---- Continual trainer tests ----

def test_trainer_step_runs():
    device = torch.device("cpu")
    model = make_model(n_classes=3).to(device)
    trainer = ContinualTrainer(model, device, cfg_path="config/network.yaml")
    feats = torch.randn(1, 63)
    labels = torch.tensor([0])
    result = trainer.step(feats, labels)
    assert "loss" in result
    assert "acc" in result
    assert isinstance(result["loss"], float)


def test_trainer_auto_expands_class():
    device = torch.device("cpu")
    model = make_model(n_classes=2).to(device)
    trainer = ContinualTrainer(model, device, cfg_path="config/network.yaml")

    # Feed class 0 first
    trainer.step(torch.randn(1, 63), torch.tensor([0]))
    assert model.n_classes == 2

    # Feed class 5 (out of range)
    trainer.step(torch.randn(1, 63), torch.tensor([5]))
    assert model.n_classes >= 6, "Model should have expanded to accommodate class 5"


def test_encode_shadow_syncs_conductances():
    """After encode_shadow, conductance decodes back close to shadow weight."""
    model = make_model(n_classes=3)
    layer = model.output_layer
    # Modify shadow weight
    with torch.no_grad():
        layer.W_shadow.fill_(0.3)
    layer.encode_shadow()
    # Decode back
    W_decoded = layer.mapper.decode(layer.G_pos, layer.G_neg)
    # Should be close to 0.3 (ideal device)
    assert (W_decoded - 0.3).abs().max() < 1e-4, \
        f"Decode mismatch: {(W_decoded - 0.3).abs().max():.2e}"
