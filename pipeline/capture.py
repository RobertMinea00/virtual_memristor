"""
Webcam capture thread.

Runs in a dedicated thread. Feeds BGR frames into a queue for the
MediaPipe / GPU thread to consume. Uses a single-slot queue (maxsize=1)
so the consumer always gets the latest frame; older frames are dropped.
"""

import threading
import queue
import cv2


class CameraCapture:
    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480):
        self._cap = cv2.VideoCapture(camera_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self) -> None:
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                continue
            # Drop old frame if consumer is slow
            if not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self._cap.release()

    def get_frame(self, timeout: float = 0.1):
        """Returns a BGR frame or None on timeout."""
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
