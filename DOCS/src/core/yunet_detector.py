"""
OpenCV YuNet Face Detector implementation using Deep Neural Network (DNN).
YuNet is a state-of-the-art lightweight (<10ms) face detector designed for CPU inference.
"""

import os
import urllib.request
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np

from src.core.base_detector import BaseDetector, DetectionResult


DEFAULT_YUNET_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)


class YuNetDetector(BaseDetector):
    """
    OpenCV YuNet DNN Face Detector.
    Detects faces with 5 key facial landmarks:
    - right_eye, left_eye, nose_tip, mouth_right, mouth_left
    """

    def __init__(
        self,
        confidence_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
        model_path: Optional[str] = None,
        auto_download: bool = True,
    ) -> None:
        super().__init__(confidence_threshold=confidence_threshold)
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.model_path = model_path or os.path.join("data", "models", "face_detection_yunet_2023mar.onnx")

        if not os.path.exists(self.model_path) and auto_download:
            self._download_model(self.model_path)

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"YuNet model file not found at {self.model_path}. "
                "Ensure auto_download=True or provide a valid path."
            )

        self._current_input_size: Tuple[int, int] = (320, 320)
        self._detector = cv2.FaceDetectorYN.create(
            model=self.model_path,
            config="",
            input_size=self._current_input_size,
            score_threshold=self.confidence_threshold,
            nms_threshold=self.nms_threshold,
            top_k=self.top_k,
        )

    def _download_model(self, target_path: str) -> None:
        os.makedirs(os.path.dirname(target_path) or ".", exist_ok=True)
        print(f"[YuNetDetector] Downloading YuNet ONNX model to {target_path}...")
        urllib.request.urlretrieve(DEFAULT_YUNET_MODEL_URL, target_path)
        print("[YuNetDetector] Download complete.")

    def set_confidence_threshold(self, threshold: float) -> None:
        super().set_confidence_threshold(threshold)
        if self._detector is not None:
            self._detector.setScoreThreshold(threshold)

    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]

        # Update input size if frame dimensions have changed
        if self._current_input_size != (w, h):
            self._current_input_size = (w, h)
            self._detector.setInputSize((w, h))

        # Perform inference
        _, faces = self._detector.detect(frame)

        results: List[DetectionResult] = []
        if faces is None or len(faces) == 0:
            return results

        for face in faces:
            # Face format: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rc, y_rc, x_lc, y_lc, score]
            origin_x = max(0, int(face[0]))
            origin_y = max(0, int(face[1]))
            box_w = min(w - origin_x, int(face[2]))
            box_h = min(h - origin_y, int(face[3]))
            score = float(face[14])

            if score < self.confidence_threshold or box_w <= 0 or box_h <= 0:
                continue

            landmarks: Dict[str, Tuple[int, int]] = {
                "right_eye": (int(face[4]), int(face[5])),
                "left_eye": (int(face[6]), int(face[7])),
                "nose_tip": (int(face[8]), int(face[9])),
                "mouth_right": (int(face[10]), int(face[11])),
                "mouth_left": (int(face[12]), int(face[13])),
            }

            results.append(
                DetectionResult(
                    bbox=(origin_x, origin_y, box_w, box_h),
                    confidence=score,
                    landmarks=landmarks,
                    extra={"detector": "OpenCV YuNet"},
                )
            )

        return results

    def close(self) -> None:
        self._detector = None
