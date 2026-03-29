"""
Main inference loop: ties capture -> landmark extraction -> GPU inference -> display.

Two-thread design:
  Thread 1 (CameraCapture): captures frames, runs MediaPipe
  Main thread: GPU inference + optional training, OpenCV display

The inference loop operates in two modes:
  - INFERENCE: just predict and display
  - COLLECT:   collect labelled samples for a class (used by collect_signs.py)
  - TRAIN:     run ContinualTrainer.step() on each incoming labelled sample
"""

import time
import queue
import threading
import numpy as np
import cv2
import torch

from pipeline.capture import CameraCapture
from pipeline.landmark_extractor import LandmarkExtractor


class InferenceLoop:
    CLASS_NAMES: list[str] = []  # populated at runtime

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        class_names: list[str] | None = None,
        trainer=None,  # ContinualTrainer | None
    ):
        self.model = model
        self.device = device
        self.trainer = trainer

        if class_names:
            InferenceLoop.CLASS_NAMES = class_names

        self._camera = CameraCapture()
        self._extractor = LandmarkExtractor()

        # Queue from extractor thread -> main thread: (features_tensor, raw_frame)
        self._feat_queue: queue.Queue = queue.Queue(maxsize=2)
        self._running = False

    # ------------------------------------------------------------------
    # Extraction thread
    # ------------------------------------------------------------------

    def _extraction_loop(self) -> None:
        while self._running:
            frame = self._camera.get_frame(timeout=0.05)
            if frame is None:
                continue
            feats = self._extractor.extract(frame)
            if feats is None:
                # No hand — push frame with None features for display
                try:
                    self._feat_queue.put_nowait((None, frame))
                except queue.Full:
                    pass
                continue
            try:
                if not self._feat_queue.empty():
                    self._feat_queue.get_nowait()
                self._feat_queue.put_nowait((feats, frame))
            except (queue.Full, queue.Empty):
                pass

    # ------------------------------------------------------------------
    # Public run method
    # ------------------------------------------------------------------

    def run(
        self,
        label_for_training: int | None = None,
        max_samples: int | None = None,
        show_window: bool = True,
    ) -> list[torch.Tensor]:
        """
        Run the loop.

        If label_for_training is set, collect that many samples for that class
        and optionally feed to trainer. Returns collected feature tensors.
        """
        self._running = True
        self._camera.start()
        ext_thread = threading.Thread(
            target=self._extraction_loop, daemon=True
        )
        ext_thread.start()

        collected: list[torch.Tensor] = []
        fps_counter, fps_t0, fps = 0, time.monotonic(), 0.0

        try:
            while self._running:
                item = self._feat_queue.get(timeout=0.1)
                feats, frame = item

                fps_counter += 1
                elapsed = time.monotonic() - fps_t0
                if elapsed >= 1.0:
                    fps = fps_counter / elapsed
                    fps_counter, fps_t0 = 0, time.monotonic()

                pred_label = None
                pred_conf = None

                if feats is not None:
                    x = feats.unsqueeze(0).to(self.device)

                    # Inference
                    self.model.eval()
                    with torch.no_grad():
                        logits = self.model(x)
                    probs = torch.softmax(logits, dim=1)
                    pred_label = probs.argmax(dim=1).item()
                    pred_conf = probs.max(dim=1).values.item()

                    # Online training
                    if label_for_training is not None and self.trainer is not None:
                        lbl = torch.tensor([label_for_training], dtype=torch.long)
                        self.trainer.step(x.cpu(), lbl)

                    # Collect
                    if label_for_training is not None:
                        collected.append(feats.cpu())
                        if max_samples and len(collected) >= max_samples:
                            break

                # Draw overlay
                if show_window and frame is not None:
                    self._draw_overlay(frame, pred_label, pred_conf, fps, feats)
                    cv2.imshow("MemristorHand", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    if key == ord("a") and label_for_training is not None:
                        label_for_training += 1  # advance to next class

        except (KeyboardInterrupt, queue.Empty):
            pass
        finally:
            self._running = False
            self._camera.stop()
            self._extractor.close()
            ext_thread.join(timeout=2.0)
            if show_window:
                cv2.destroyAllWindows()

        return collected

    def stop(self) -> None:
        self._running = False

    @staticmethod
    def _draw_overlay(
        frame: np.ndarray,
        pred_label: int | None,
        conf: float | None,
        fps: float,
        feats,
    ) -> None:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 0), -1)

        if pred_label is not None and conf is not None:
            names = InferenceLoop.CLASS_NAMES
            name = names[pred_label] if pred_label < len(names) else f"cls{pred_label}"
            text = f"{name}  ({conf:.1%})"
            cv2.putText(frame, text, (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 100), 2)
        else:
            cv2.putText(frame, "No hand detected", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)

        cv2.putText(frame, f"{fps:.0f} FPS", (w - 100, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

        if feats is not None:
            # Small skeleton visualisation
            pts_2d = feats.reshape(21, 3)[:, :2].numpy()
            pts_2d = (pts_2d * 0.3 + 0.5).clip(0, 1)  # normalise to frame
            for j, pt in enumerate(pts_2d):
                cx, cy = int(pt[0] * w), int(pt[1] * h)
                cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
