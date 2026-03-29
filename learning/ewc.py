"""
Elastic Weight Consolidation (EWC) regulariser.

After learning each task, computes the diagonal Fisher information matrix
for all shadow parameters and stores it as an importance estimate.
The EWC penalty prevents important weights from changing too much.

Reference: Kirkpatrick et al. 2017, "Overcoming catastrophic forgetting
in neural networks".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


class EWC:
    def __init__(self, model: nn.Module, lambda_: float = 400.0):
        self.model = model
        self.lambda_ = lambda_

        # {param_name: (mean, fisher_diag)} accumulated across all tasks
        self._consolidated: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    @torch.no_grad()
    def _get_named_shadow_params(self) -> dict[str, torch.Tensor]:
        """Return all shadow weight parameters by name."""
        params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params[name] = param
        return params

    def consolidate(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        n_samples: int = 200,
    ) -> None:
        """
        Compute and store Fisher diagonal for current task.
        Call this after the model has learned a task, before introducing new data.

        features: (N, feature_dim) — representative samples from current task(s)
        labels:   (N,) long tensor
        """
        self.model.eval()
        device = features.device

        # Subsample for efficiency
        n = min(n_samples, len(features))
        idx = torch.randperm(len(features))[:n]
        feat_sub = features[idx].to(device)
        lbl_sub = labels[idx].to(device)

        # Accumulate squared gradients (Fisher diagonal estimate)
        fisher: dict[str, torch.Tensor] = {}
        named_params = self._get_named_shadow_params()

        for name in named_params:
            fisher[name] = torch.zeros_like(named_params[name])

        self.model.zero_grad()
        for i in range(n):
            x = feat_sub[i:i+1]
            y = lbl_sub[i:i+1]
            logits = self.model(x)
            # Use log-likelihood of predicted class (standard Fisher approx)
            log_prob = F.log_softmax(logits, dim=1)
            loss = F.nll_loss(log_prob, y)
            loss.backward()

            for name, param in named_params.items():
                if param.grad is not None:
                    fisher[name] += param.grad.detach() ** 2

            self.model.zero_grad()

        # Normalise and accumulate
        for name in fisher:
            fisher[name] /= n
            mean = named_params[name].detach().clone()
            if name in self._consolidated:
                # The output layer may have grown since last consolidation.
                # Zero-pad the old Fisher to match current param shape so that
                # new neurons have no consolidation penalty (they're unconstrained).
                old_mean, old_fisher = self._consolidated[name]
                new_shape = fisher[name].shape
                if old_fisher.shape != new_shape:
                    padded_fisher = torch.zeros(new_shape, device=old_fisher.device)
                    slices = tuple(slice(0, s) for s in old_fisher.shape)
                    padded_fisher[slices] = old_fisher
                    old_fisher = padded_fisher
                self._consolidated[name] = (mean, old_fisher + fisher[name])
            else:
                self._consolidated[name] = (mean, fisher[name])

        self.model.train()

    def penalty(self) -> torch.Tensor:
        """
        Compute the EWC regularisation loss term.
        Add `lambda_ * penalty()` to the cross-entropy loss during training.
        """
        if not self._consolidated:
            return torch.tensor(0.0)

        loss = torch.tensor(0.0)
        named_params = self._get_named_shadow_params()

        for name, (mean, fisher) in self._consolidated.items():
            if name in named_params:
                param = named_params[name]
                loss = loss + (fisher * (param - mean.to(param.device)) ** 2).sum()

        return self.lambda_ * loss
