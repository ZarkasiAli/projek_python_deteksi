"""
FPS and Latency Calculation module using moving-average time windows.
Provides accurate framerate tracking and sub-millisecond inference latency measurement.
"""

import time
from collections import deque
from typing import Optional


class FPSCounter:
    """
    Tracks and computes real-time frames per second (FPS) and inference latency.
    """

    def __init__(self, window_size: int = 30) -> None:
        self.window_size = window_size
        self._timestamps = deque(maxlen=window_size)
        self._latency_history = deque(maxlen=window_size)
        self._inference_start_time: Optional[float] = None
        self._last_fps: float = 0.0
        self._last_latency_ms: float = 0.0

    def tick(self) -> None:
        """Call at the start of a frame processing cycle."""
        self._timestamps.append(time.perf_counter())

    def start_inference(self) -> None:
        """Record the start timestamp of AI model inference."""
        self._inference_start_time = time.perf_counter()

    def end_inference(self) -> float:
        """
        Record the end of AI model inference and update moving-average latency.
        
        Returns:
            Current inference latency in milliseconds (ms).
        """
        if self._inference_start_time is None:
            return 0.0
        latency_ms = (time.perf_counter() - self._inference_start_time) * 1000.0
        self._latency_history.append(latency_ms)
        self._last_latency_ms = latency_ms
        self._inference_start_time = None
        return latency_ms

    @property
    def fps(self) -> float:
        """Calculate the average FPS over the configured window."""
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed

    @property
    def latency_ms(self) -> float:
        """Return the average inference latency in milliseconds."""
        if not self._latency_history:
            return 0.0
        return sum(self._latency_history) / len(self._latency_history)

    def reset(self) -> None:
        """Clear recorded timestamps and latency metrics."""
        self._timestamps.clear()
        self._latency_history.clear()
        self._inference_start_time = None
