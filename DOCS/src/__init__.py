"""
Face Detection and Tracking System Root Package.
"""

from src.core import BaseDetector, DetectionResult, MediaPipeDetector, YuNetDetector, create_detector
from src.video import FPSCounter, ThreadedVideoStream
from src.tracking import FaceTracker
from src.utils import Visualizer, DetectionLogger, SnapshotManager
from src.config import AppConfig

__all__ = [
    "BaseDetector",
    "DetectionResult",
    "MediaPipeDetector",
    "YuNetDetector",
    "create_detector",
    "FPSCounter",
    "ThreadedVideoStream",
    "FaceTracker",
    "Visualizer",
    "DetectionLogger",
    "SnapshotManager",
    "AppConfig",
]
