"""
High-performance Producer-Consumer Threaded Video Stream.
Decouples camera I/O and frame decoding from model inference to prevent capture lag and latency spikes.
"""

import threading
import time
from typing import Union, Optional, Tuple
import cv2
import numpy as np


class ThreadedVideoStream:
    """
    Asynchronous video stream that reads frames in a dedicated background thread.
    
    Prevents OpenCV `cap.read()` buffer lag by always maintaining the newest frame available.
    """

    def __init__(
        self,
        src: Union[int, str] = 0,
        name: str = "ThreadedVideoStream",
        width: Optional[int] = None,
        height: Optional[int] = None,
        target_fps: Optional[int] = None,
    ) -> None:
        self.src = src
        self.name = name
        self.target_width = width
        self.target_height = height
        self.target_fps = target_fps

        self.stream = cv2.VideoCapture(self.src)
        if self.target_width is not None:
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        if self.target_height is not None:
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
        if self.target_fps is not None:
            self.stream.set(cv2.CAP_PROP_FPS, self.target_fps)

        # Warm-up grab
        self.grabbed, self.frame = self.stream.read()
        self.is_opened = self.stream.isOpened()

        self._stopped = False
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "ThreadedVideoStream":
        """Start the background frame grabbing thread."""
        if self._stopped or not self.is_opened:
            return self

        self._stopped = False
        self._thread = threading.Thread(target=self._update, name=self.name, daemon=True)
        self._thread.start()
        return self

    def _update(self) -> None:
        """Producer loop: continuously reads frames from stream into thread-safe buffer."""
        while not self._stopped:
            if not self.stream.isOpened():
                with self._lock:
                    self.grabbed = False
                break

            grabbed, frame = self.stream.read()

            with self._lock:
                self.grabbed = grabbed
                if grabbed:
                    self.frame = frame
                else:
                    # Stream ended or camera disconnected
                    self._stopped = True
                    break

            # Micro-sleep to prevent 100% single-core spin on fast streams
            time.sleep(0.001)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Consumer read: safely retrieves the latest decoded frame.
        
        Returns:
            Tuple of (status_boolean, frame_array).
        """
        with self._lock:
            if not self.grabbed or self.frame is None:
                return False, None
            return True, self.frame.copy()

    @property
    def resolution(self) -> Tuple[int, int]:
        """Return (width, height) of the active stream."""
        with self._lock:
            if self.frame is not None:
                h, w = self.frame.shape[:2]
                return (w, h)
        w = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)

    def stop(self) -> None:
        """Signal thread to stop and release video capture resources."""
        self._stopped = True
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self.stream.isOpened():
            self.stream.release()

    def release(self) -> None:
        """Alias for stop()."""
        self.stop()

    def __enter__(self) -> "ThreadedVideoStream":
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
