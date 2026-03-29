"""
Online continual learning trainer.

Processes one sample (or a small batch) at a time as they arrive
from the webcam or benchmark data stream.

Learning strategy (each independently toggleable via config):
  1. Online cross-entropy update on current sample
  2. Episodic replay from the GPU-resident buffer
  3. EWC regularisation penalty
  4. Conductance write suppression for high-importance weights (hardware-native)
"""

import time
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from network.classifier import MemristorClassifier
from learning.replay_buffer import ReplayBuffer
from learning.ewc import EWC
from learning.class_expansion import expand_output_layer


class ContinualTrainer:
    def __init__(
        self,
        model: MemristorClassifier,
        device: torch.device,
        cfg_path: str = "config/network.yaml",
    ):
        self.model = model
        self.device = device

        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        tcfg = cfg["training"]

        self.lr = tcfg["lr"]
        self.online_batch = tcfg["online_batch_size"]
        self.replay_batch = tcfg["replay_batch_size"]
        self.use_replay = tcfg["use_replay"]
        self.use_ewc = tcfg["use_ewc"]
        self.ewc_lambda = tcfg["ewc_lambda"]

        self.buffer = ReplayBuffer(
            capacity=tcfg["replay_buffer_size"],
            feature_dim=cfg["input_dim"],
            device=device,
        )
        self.ewc = EWC(model, lambda_=self.ewc_lambda)

        # Use Adam on shadow weights only
        self.optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=self.lr,
        )

        self.known_classes: set[int] = set()
        self._model_lock = threading.Lock()
        self._n_samples_seen = 0

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def step(
        self,
        features: torch.Tensor,  # (1, 63) or (B, 63)
        labels: torch.Tensor,    # (1,) or (B,) long
    ) -> dict:
        """
        Process one online sample (or small batch).
        Returns a metrics dict: {loss, acc, new_class_added}.
        """
        features = features.to(self.device)
        labels = labels.to(self.device)
        self._n_samples_seen += len(labels)

        # Check for new classes
        new_class_added = False
        for label in labels.tolist():
            if label not in self.known_classes:
                new_class_added = True
                # Expand until the model has enough output neurons
                while label >= self.model.n_classes:
                    expand_output_layer(self.model, self._model_lock)
                self.known_classes.add(label)

        # If we just expanded the model, consolidate EWC on the old tasks
        if new_class_added and len(self.known_classes) > 1 and self.use_ewc:
            if len(self.buffer) >= 50:
                buf_f, buf_l = self.buffer.sample(200)
                self.ewc.consolidate(buf_f, buf_l)

        with self._model_lock:
            self.model.train()
            self.optimizer.zero_grad()

            # Online loss on current sample
            logits = self.model(features)
            loss = F.cross_entropy(logits, labels)

            # Replay loss
            if self.use_replay and len(self.buffer) >= self.replay_batch:
                rep_f, rep_l = self.buffer.sample(self.replay_batch)
                rep_logits = self.model(rep_f)
                loss = loss + F.cross_entropy(rep_logits, rep_l)

            # EWC penalty
            if self.use_ewc:
                loss = loss + self.ewc.penalty().to(self.device)

            loss.backward()
            self.optimizer.step()

            # Project shadow weights -> conductances
            self.model.encode_all_shadows()

        # Add current sample to replay buffer
        with torch.no_grad():
            self.buffer.add_batch(features.cpu(), labels.cpu())

        # Accuracy on current sample (no noise for metric eval)
        with torch.no_grad():
            self.model.eval()
            pred = self.model(features).argmax(dim=1)
            acc = (pred == labels).float().mean().item()

        return {
            "loss": loss.item(),
            "acc": acc,
            "new_class_added": new_class_added,
            "n_classes": self.model.n_classes,
            "buffer_size": len(self.buffer),
        }

    def eval_classes(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        disable_noise: bool = True,
    ) -> dict[int, float]:
        """
        Evaluate per-class accuracy on a held-out set.
        Returns {class_idx: accuracy}.
        """
        features = features.to(self.device)
        labels = labels.to(self.device)

        orig_noise = [l.apply_noise for l in self.model.memristor_layers()]
        if disable_noise:
            for l in self.model.memristor_layers():
                l.apply_noise = False

        self.model.eval()
        with torch.no_grad():
            preds = self.model(features).argmax(dim=1)

        if disable_noise:
            for l, n in zip(self.model.memristor_layers(), orig_noise):
                l.apply_noise = n

        per_class: dict[int, float] = {}
        for cls in labels.unique().tolist():
            mask = labels == cls
            per_class[int(cls)] = (preds[mask] == labels[mask]).float().mean().item()
        return per_class
