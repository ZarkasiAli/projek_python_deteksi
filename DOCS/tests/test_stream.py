"""
Unit tests for Video Streaming, FPSCounter, and Configuration.
"""

import time
import numpy as np
import pytest
from src.video.fps_counter import FPSCounter
from src.video.video_stream import ThreadedVideoStream
from src.config import AppConfig


def test_fps_counter_calculation():
    counter = FPSCounter(window_size=10)
    for _ in range(5):
        counter.tick()
        time.sleep(0.01)

    assert counter.fps > 0.0

    counter.start_inference()
    time.sleep(0.005)
    lat = counter.end_inference()

    assert lat > 0.0
    assert counter.latency_ms > 0.0

    counter.reset()
    assert counter.fps == 0.0
    assert counter.latency_ms == 0.0


def test_app_config_initialization():
    config = AppConfig()
    assert config.detector.model_type == "mediapipe"
    assert config.tracking.enabled is True
    assert config.stream.target_fps == 30
    config.ensure_directories()


def test_video_stream_initialization_invalid_source():
    # Attempting to open a non-existent video file should handle gracefully
    stream = ThreadedVideoStream(src="non_existent_video_path_123.mp4")
    assert stream.is_opened is False
    status, frame = stream.read()
    assert status is False
    assert frame is None
    stream.stop()
