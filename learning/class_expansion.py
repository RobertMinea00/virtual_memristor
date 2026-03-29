"""
Live class expansion — add a new output class to the classifier
without disturbing existing neurons.
"""

import torch
import threading
from network.classifier import MemristorClassifier


def expand_output_layer(model: MemristorClassifier, lock: threading.Lock) -> int:
    """
    Thread-safe expansion of the output layer by one class.

    Acquires lock to pause inference, expands, releases.
    Returns the index of the new class.
    """
    with lock:
        new_class_idx = model.add_class()
    return new_class_idx
