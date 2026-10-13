"""
Abstract Base Class and data structures for face detectors.
Follows SOLID principles for pluggable computer vision detectors.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np


@dataclass
class DetectionResult:
    """
    Structured representation of a single detected face.
    
    Attributes:
        bbox: Bounding box tuple (x, y, width, height) in integer pixel coordinates.
        confidence: Detection confidence score between 0.0 and 1.0.
        landmarks: Dictionary of facial landmark points (e.g. 'right_eye', 'left_eye', 'nose_tip', etc.).
        track_id: Optional persistent tracking identifier.
        extra: Additional metadata (e.g. angles, quality score).
    """
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float
    landmarks: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    track_id: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def x(self) -> int:
        return self.bbox[0]

    @property
    def y(self) -> int:
        return self.bbox[1]

    @property
    def w(self) -> int:
        return self.bbox[2]

    @property
    def h(self) -> int:
        return self.bbox[3]

    @property
    def x2(self) -> int:
        return self.bbox[0] + self.bbox[2]

    @property
    def y2(self) -> int:
        return self.bbox[1] + self.bbox[3]

    @property
    def center(self) -> Tuple[int, int]:
        return (self.bbox[0] + self.bbox[2] // 2, self.bbox[1] + self.bbox[3] // 2)

    @property
    def area(self) -> int:
        return self.bbox[2] * self.bbox[3]


class BaseDetector(ABC):
    """
    Abstract Base Class for all face detection models.
    Enforces uniform interface across MediaPipe, YuNet, YOLOv8, etc.
    """

    def __init__(self, confidence_threshold: float = 0.6) -> None:
        if not (0.0 <= confidence_threshold <= 1.0):
            raise ValueError(f"Confidence threshold must be between 0.0 and 1.0, got {confidence_threshold}")
        self.confidence_threshold = confidence_threshold

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        """
        Execute face detection on a single BGR or RGB video frame.

        Args:
            frame: Input image array (H, W, C) in uint8 format.

        Returns:
            List of DetectionResult instances for all detected faces.
        """
        pass

    def set_confidence_threshold(self, threshold: float) -> None:
        """Update detection confidence threshold dynamically."""
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"Confidence threshold must be between 0.0 and 1.0, got {threshold}")
        self.confidence_threshold = threshold

    def close(self) -> None:
        """Clean up underlying model resources if necessary."""
        pass

    def __enter__(self) -> "BaseDetector":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
