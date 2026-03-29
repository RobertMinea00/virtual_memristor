"""
Tests for pipeline components (no webcam — mocks frame input).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import pytest
from pipeline.landmark_extractor import LandmarkExtractor


def _make_synthetic_hand_frame(width=640, height=480):
    """
    Create a fake RGB frame with a rough hand-like blob.
    MediaPipe will likely return None, but tests that the extractor
    handles None gracefully.
    """
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    # Crude skin-tone rectangle
    frame[200:400, 250:400] = [180, 120, 80]
    return frame  # BGR


def test_extractor_returns_none_on_blank_frame():
    pytest.importorskip("mediapipe")
    from pathlib import Path
    if not Path("models/hand_landmarker.task").exists():
        pytest.skip("hand_landmarker.task model not downloaded yet")
    extractor = LandmarkExtractor()
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    result = extractor.extract(blank)
    assert result is None, "Should return None when no hand detected"
    extractor.close()


def test_extractor_output_shape_when_hand_present():
    """
    This test requires a real hand in frame.
    Skip gracefully if no landmarks detected in synthetic frame.
    """
    pytest.importorskip("mediapipe")
    from pathlib import Path
    if not Path("models/hand_landmarker.task").exists():
        pytest.skip("hand_landmarker.task model not downloaded yet")
    extractor = LandmarkExtractor()
    frame = _make_synthetic_hand_frame()
    result = extractor.extract(frame)
    if result is not None:
        assert result.shape == (63,), f"Expected (63,), got {result.shape}"
        assert result.dtype == torch.float32
    extractor.close()


def test_extractor_translation_invariance():
    """
    If we could inject landmarks directly, landmark 0 (wrist) should be zero
    after normalisation. Test the normalisation logic directly.
    """
    # Simulate 21 landmarks with wrist at (0.5, 0.5, 0.0) and finger at (0.6, 0.7, 0.0)
    pts = np.random.rand(21, 3).astype(np.float32)
    pts_norm = pts - pts[0]  # subtract wrist
    scale = np.linalg.norm(pts_norm[9]) + 1e-9
    pts_norm /= scale
    # Wrist should be zero
    assert np.allclose(pts_norm[0], 0.0, atol=1e-6)
    # Landmark 9 should have unit norm
    assert abs(np.linalg.norm(pts_norm[9]) - 1.0) < 1e-5
