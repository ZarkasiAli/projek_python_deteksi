"""
Unit tests for Core Face Detectors (MediaPipe & YuNet).
"""

import numpy as np
import pytest
from src.core.base_detector import BaseDetector, DetectionResult
from src.core.mediapipe_detector import MediaPipeDetector
from src.core.yunet_detector import YuNetDetector
from src.core.detector_factory import create_detector


def test_detection_result_properties():
    result = DetectionResult(
        bbox=(10, 20, 100, 120),
        confidence=0.85,
        landmarks={"right_eye": (40, 50), "left_eye": (80, 50)},
    )
    assert result.x == 10
    assert result.y == 20
    assert result.w == 100
    assert result.h == 120
    assert result.x2 == 110
    assert result.y2 == 140
    assert result.center == (60, 80)
    assert result.area == 12000


def test_mediapipe_detector_empty_frame():
    detector = MediaPipeDetector(confidence_threshold=0.5)
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    results = detector.detect(blank)
    assert isinstance(results, list)
    assert len(results) == 0
    detector.close()


def test_yunet_detector_empty_frame():
    detector = YuNetDetector(confidence_threshold=0.5)
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    results = detector.detect(blank)
    assert isinstance(results, list)
    assert len(results) == 0
    detector.close()


def test_detector_factory():
    mp_det = create_detector("mediapipe", confidence_threshold=0.7)
    assert isinstance(mp_det, MediaPipeDetector)
    assert mp_det.confidence_threshold == 0.7
    mp_det.close()

    yu_det = create_detector("yunet", confidence_threshold=0.65)
    assert isinstance(yu_det, YuNetDetector)
    assert yu_det.confidence_threshold == 0.65
    yu_det.close()

    with pytest.raises(ValueError):
        create_detector("unknown_detector")


def test_confidence_threshold_update():
    detector = YuNetDetector(confidence_threshold=0.5)
    detector.set_confidence_threshold(0.8)
    assert detector.confidence_threshold == 0.8

    with pytest.raises(ValueError):
        detector.set_confidence_threshold(1.5)
    detector.close()
