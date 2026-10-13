"""
Factory for creating face detectors dynamically based on model selection.
"""

from typing import Optional
from src.core.base_detector import BaseDetector
from src.core.mediapipe_detector import MediaPipeDetector
from src.core.yunet_detector import YuNetDetector


def create_detector(
    model_type: str = "mediapipe",
    confidence_threshold: float = 0.6,
    model_path: Optional[str] = None,
    **kwargs
) -> BaseDetector:
    """
    Factory helper to instantiate a face detector.

    Args:
        model_type: "mediapipe" (default) or "yunet".
        confidence_threshold: Minimum detection confidence threshold (0.0 to 1.0).
        model_path: Custom path to model weight file (optional).

    Returns:
        Instance of BaseDetector.
    """
    model_key = model_type.strip().lower()
    if model_key in ("mediapipe", "blazeface"):
        return MediaPipeDetector(
            confidence_threshold=confidence_threshold,
            model_path=model_path,
            **kwargs
        )
    elif model_key in ("yunet", "opencv", "dnn"):
        return YuNetDetector(
            confidence_threshold=confidence_threshold,
            model_path=model_path,
            **kwargs
        )
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Supported detectors: 'mediapipe', 'yunet'."
        )
