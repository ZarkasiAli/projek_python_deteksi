"""
Unit tests for Multi-Face Tracker and Visualizer/Logger utilities.
"""

import os
import shutil
import numpy as np
import pytest
from src.core.base_detector import DetectionResult
from src.tracking.tracker import FaceTracker
from src.utils.visualizer import Visualizer
from src.utils.logger import DetectionLogger
from src.utils.snapshot import SnapshotManager
from src.config import LoggingConfig, SnapshotConfig


def test_tracker_registration_and_update():
    tracker = FaceTracker(max_disappeared=5)

    # Frame 1: One face detected
    det1 = DetectionResult(bbox=(50, 50, 100, 100), confidence=0.9)
    res1 = tracker.update([det1])

    assert len(res1) == 1
    assert res1[0].track_id == 1

    # Frame 2: Face moved slightly (IoU match)
    det2 = DetectionResult(bbox=(55, 52, 100, 100), confidence=0.88)
    res2 = tracker.update([det2])

    assert len(res2) == 1
    assert res2[0].track_id == 1  # Retains ID #1

    # Frame 3: Second face appears elsewhere
    det3_a = DetectionResult(bbox=(58, 54, 100, 100), confidence=0.87)
    det3_b = DetectionResult(bbox=(300, 200, 80, 80), confidence=0.92)
    res3 = tracker.update([det3_a, det3_b])

    assert len(res3) == 2
    assert res3[0].track_id == 1
    assert res3[1].track_id == 2  # New ID #2


def test_tracker_disappearance_and_deregistration():
    tracker = FaceTracker(max_disappeared=2)

    det = DetectionResult(bbox=(10, 10, 50, 50), confidence=0.8)
    tracker.update([det])
    assert 1 in tracker.tracked_objects

    # 1st missed frame
    tracker.update([])
    assert 1 in tracker.tracked_objects
    assert tracker.tracked_objects[1].disappeared_count == 1

    # 2nd missed frame
    tracker.update([])
    assert 1 in tracker.tracked_objects
    assert tracker.tracked_objects[1].disappeared_count == 2

    # 3rd missed frame (exceeds max_disappeared=2 -> deregister)
    tracker.update([])
    assert 1 not in tracker.tracked_objects


def test_visualizer_rendering():
    viz = Visualizer()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    det = DetectionResult(
        bbox=(100, 100, 150, 180),
        confidence=0.92,
        landmarks={"right_eye": (130, 140), "left_eye": (210, 140)},
        track_id=1,
    )
    rendered = viz.draw(frame, [det], fps=30.0, latency_ms=12.5, model_name="MediaPipe")

    assert rendered.shape == frame.shape
    assert not np.array_equal(rendered, frame)  # Overlay should modify image pixels


def test_snapshot_manager(tmp_path):
    config = SnapshotConfig(output_dir=str(tmp_path / "snapshots"))
    mgr = SnapshotManager(config)

    frame = np.ones((400, 400, 3), dtype=np.uint8) * 128
    det = DetectionResult(bbox=(50, 50, 100, 100), confidence=0.9, track_id=5)

    path = mgr.save_face_snapshot(frame, det)
    assert path is not None
    assert os.path.exists(path)


def test_detection_logger(tmp_path):
    config = LoggingConfig(log_dir=str(tmp_path / "logs"), export_format="both")
    logger = DetectionLogger(config)

    det = DetectionResult(
        bbox=(20, 30, 80, 90),
        confidence=0.85,
        landmarks={"nose": (60, 70)},
        track_id=3,
    )
    logger.log_detection(frame_idx=1, detection=det, model_name="YuNet")
    logger.close()

    assert os.path.exists(logger.csv_path)
    assert os.path.exists(logger.jsonl_path)
