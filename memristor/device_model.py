"""
Memristor device physics engine.

All operations are pure PyTorch tensor ops that run on whatever device
the input tensors live on (CPU or CUDA). No .cpu() calls here.

Five non-ideal effects, each independently toggleable via the config:
  1. Conductance quantisation   (write-time)
  2. Write noise                (write-time)
  3. Device-to-device variability (init-time, fixed)
  4. Conductance drift          (time-dependent)
  5. Read noise                 (inference-time)
"""

import torch
import torch.nn as nn


class MemristorDeviceModel:
    """
    Stateless helper — contains only the physics parameters.
    Stateful tensors (G_pos, G_neg, t_last_written, d2d offsets)
    live in the MemristorLinear module that owns them.
    """

    def __init__(self, cfg: dict):
        # Support both nested YAML format (keys under "device:") and flat test dicts
        dev = cfg.get("device", cfg)
        self.g_min = dev["g_min"]
        self.g_max = dev["g_max"]
        self.g_range = self.g_max - self.g_min
        self.n_levels = dev.get("n_levels", cfg.get("n_levels", 0))
        self.sigma_write = cfg["noise"]["sigma_write"]
        self.sigma_read = cfg["noise"]["sigma_read"]
        self.sigma_d2d = cfg["noise"]["sigma_d2d"]
        self.nu = cfg["drift"]["nu"]
        self.t0 = cfg["drift"]["t0"]
        self.drift_every_n = cfg["drift"]["update_every_n"]

    # ------------------------------------------------------------------
    # 1 & 2: Quantisation + write noise  (applied after each weight update)
    # ------------------------------------------------------------------

    def program(self, G_target: torch.Tensor) -> torch.Tensor:
        """
        Clamp to [g_min, g_max], quantise to n_levels, then add write noise.
        Returns the physically realised conductance after programming.
        """
        G = G_target.clamp(self.g_min, self.g_max)

        # Quantise
        if self.n_levels > 0:
            # Normalise to [0, n_levels-1], round, denormalise
            G_norm = (G - self.g_min) / self.g_range
            G_q = (G_norm * (self.n_levels - 1)).round() / (self.n_levels - 1)
            G = self.g_min + G_q * self.g_range

        # Write noise: sigma proportional to the conductance step size
        if self.sigma_write > 0.0:
            delta_g = self.g_range / max(self.n_levels - 1, 1)
            noise = torch.randn_like(G) * (self.sigma_write * delta_g)
            G = (G + noise).clamp(self.g_min, self.g_max)

        return G

    # ------------------------------------------------------------------
    # 3: Device-to-device variability  (call once at layer init)
    # ------------------------------------------------------------------

    def init_d2d_offset(self, shape: tuple, device: torch.device) -> torch.Tensor:
        """
        Fixed per-device multiplicative offset drawn from N(1, sigma_d2d).
        Store as a non-trainable buffer; apply to G before every read.
        """
        if self.sigma_d2d > 0.0:
            return torch.empty(shape, device=device).normal_(1.0, self.sigma_d2d)
        return torch.ones(shape, device=device)

    # ------------------------------------------------------------------
    # 4: Drift  (power-law, applied periodically)
    # ------------------------------------------------------------------

    def apply_drift(
        self,
        G: torch.Tensor,
        G_at_write: torch.Tensor,
        elapsed_seconds: float,
    ) -> torch.Tensor:
        """
        G(t) = G_write * (elapsed / t0) ^ -nu
        Only applied when elapsed > t0 to avoid negative exponent oddities.
        """
        if self.nu == 0.0 or elapsed_seconds <= self.t0:
            return G
        factor = (elapsed_seconds / self.t0) ** (-self.nu)
        return (G_at_write * factor).clamp(self.g_min, self.g_max)

    # ------------------------------------------------------------------
    # 5: Read noise  (applied every forward pass during training)
    # ------------------------------------------------------------------

    def apply_read_noise(self, G: torch.Tensor, d2d: torch.Tensor) -> torch.Tensor:
        """
        G_read = G_nominal * d2d_offset * (1 + N(0, sigma_read))
        d2d is the fixed per-device offset tensor from init_d2d_offset.
        """
        G_with_d2d = G * d2d
        if self.sigma_read > 0.0:
            noise = torch.randn_like(G_with_d2d) * self.sigma_read
            return G_with_d2d * (1.0 + noise)
        return G_with_d2d

    # ------------------------------------------------------------------
    # Straight-through estimator for quantisation in backward pass
    # ------------------------------------------------------------------

    @staticmethod
    def ste_round(x: torch.Tensor) -> torch.Tensor:
        """Round with straight-through gradient estimator."""
        return x + (x.round() - x).detach()
