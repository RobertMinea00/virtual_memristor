"""
MediaPipe Hands wrapper — uses the Tasks API (mediapipe >= 0.10.21).

On first use, downloads hand_landmarker.task to models/hand_landmarker.task
if it is not already present.

Extracts 21 3D landmarks from a BGR frame, normalises to be
translation- and scale-invariant, returns a (63,) float32 CPU tensor.
"""

import os
import urllib.request
from pathlib import Path

import numpy as np
import torch

# Model URL (Google's official hosted model)
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
_MODEL_PATH = Path("models/hand_landmarker.task")


def _ensure_model() -> str:
    if not _MODEL_PATH.exists():
        _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading hand landmarker model to {_MODEL_PATH} …")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("Download complete.")
    return str(_MODEL_PATH)


class LandmarkExtractor:
    def __init__(
        self,
        max_hands: int = 1,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        model_path: str | None = None,
    ):
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision

        if model_path is None:
            model_path = _ensure_model()

        base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._timestamp_ms = 0

    def extract(self, frame_bgr: np.ndarray) -> torch.Tensor | None:
        """
        Process one BGR frame. Returns a (63,) float32 tensor on CPU,
        or None if no hand is detected.
        """
        import cv2
        from mediapipe.framework.formats import landmark_pb2
        import mediapipe as mp

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self._timestamp_ms += 33  # ~30 FPS timestamps
        result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)

        if not result.hand_landmarks:
            return None

        lm = result.hand_landmarks[0]  # first hand
        pts = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)  # (21, 3)

        # Translation invariance: subtract wrist (landmark 0)
        pts -= pts[0]

        # Scale invariance: divide by wrist-to-middle-MCP distance (landmark 9)
        scale = np.linalg.norm(pts[9]) + 1e-9
        pts /= scale

        return torch.from_numpy(pts.flatten())  # (63,)

    def close(self) -> None:
        self._landmarker.close()
