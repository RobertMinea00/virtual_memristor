"""
MediaPipe Hands wrapper — Tasks API (mediapipe >= 0.10.21).

Downloads hand_landmarker.task to models/ on first use.
Returns a (63,) float32 tensor: 21 landmarks * 3 coords,
translation- and scale-normalised. Returns None if no hand found.
"""

import urllib.request
from pathlib import Path

import numpy as np
import torch

_MODEL_URL  = (
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
        min_detection_confidence: float = 0.5,   # lowered from 0.7
        model_path: str | None = None,
    ):
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision

        if model_path is None:
            model_path = _ensure_model()

        base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
        # IMAGE mode: stateless per-frame, no timestamp needed
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=0.4,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)

    def extract(self, frame_bgr: np.ndarray) -> torch.Tensor | None:
        """
        Process one BGR frame.
        Returns (63,) float32 tensor on CPU, or None if no hand detected.
        """
        import cv2
        import mediapipe as mp

        rgb      = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = self._landmarker.detect(mp_image)   # IMAGE mode

        if not result.hand_landmarks:
            return None

        lm  = result.hand_landmarks[0]
        pts = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)  # (21,3)

        pts -= pts[0]                                   # translation invariant
        scale = np.linalg.norm(pts[9]) + 1e-9
        pts  /= scale                                   # scale invariant

        return torch.from_numpy(pts.flatten())          # (63,)

    def close(self) -> None:
        self._landmarker.close()
