"""
MediaPipe Face Detection implementation using Google's BlazeFace model.
Provides ultra-low latency (<15ms on modern CPU) and robust multi-face detection with 6 landmarks.
"""

import os
import urllib.request
from typing import List, Dict, Tuple, Optional
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src.core.base_detector import BaseDetector, DetectionResult


DEFAULT_MEDIAPIPE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
)


class MediaPipeDetector(BaseDetector):
    """
    BlazeFace detector using MediaPipe Tasks Vision API.
    Detects faces with 6 key landmarks:
    - right_eye, left_eye, nose_tip, mouth_center, right_ear_tragion, left_ear_tragion
    """

    def __init__(
        self,
        confidence_threshold: float = 0.6,
        model_path: Optional[str] = None,
        auto_download: bool = True,
    ) -> None:
        super().__init__(confidence_threshold=confidence_threshold)
        
        self.model_path = model_path or os.path.join("data", "models", "blaze_face_short_range.tflite")
        if not os.path.exists(self.model_path) and auto_download:
            self._download_model(self.model_path)
            
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"MediaPipe model file not found at {self.model_path}. "
                "Ensure auto_download=True or provide a valid path."
            )

        self._detector: Optional[vision.FaceDetector] = None
        self._init_detector()

    def _download_model(self, target_path: str) -> None:
        os.makedirs(os.path.dirname(target_path) or ".", exist_ok=True)
        print(f"[MediaPipeDetector] Downloading BlazeFace model to {target_path}...")
        urllib.request.urlretrieve(DEFAULT_MEDIAPIPE_MODEL_URL, target_path)
        print("[MediaPipeDetector] Download complete.")

    def _init_detector(self) -> None:
        if self._detector is not None:
            self._detector.close()

        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=self.confidence_threshold,
            running_mode=vision.RunningMode.IMAGE,
        )
        self._detector = vision.FaceDetector.create_from_options(options)

    def set_confidence_threshold(self, threshold: float) -> None:
        super().set_confidence_threshold(threshold)
        self._init_detector()

    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        # MediaPipe expects RGB format
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            rgb_frame = frame[:, :, ::-1].copy()  # BGR to RGB
        else:
            rgb_frame = frame

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self._detector.detect(mp_image)

        results: List[DetectionResult] = []
        if not detection_result.detections:
            return results

        landmark_names = [
            "right_eye",
            "left_eye",
            "nose_tip",
            "mouth_center",
            "right_ear",
            "left_ear",
        ]

        for det in detection_result.detections:
            # Score
            score = det.categories[0].score if det.categories else 0.0
            if score < self.confidence_threshold:
                continue

            # Bounding box
            bb = det.bounding_box
            origin_x = max(0, int(bb.origin_x))
            origin_y = max(0, int(bb.origin_y))
            box_w = min(w - origin_x, int(bb.width))
            box_h = min(h - origin_y, int(bb.height))

            if box_w <= 0 or box_h <= 0:
                continue

            # Keypoints / Landmarks
            landmarks: Dict[str, Tuple[int, int]] = {}
            if det.keypoints:
                for idx, kp in enumerate(det.keypoints):
                    name = landmark_names[idx] if idx < len(landmark_names) else f"point_{idx}"
                    px = int(kp.x * w)
                    py = int(kp.y * h)
                    landmarks[name] = (px, py)

            results.append(
                DetectionResult(
                    bbox=(origin_x, origin_y, box_w, box_h),
                    confidence=float(score),
                    landmarks=landmarks,
                    extra={"detector": "MediaPipe BlazeFace"},
                )
            )

        return results

    def close(self) -> None:
        if self._detector is not None:
            self._detector.close()
            self._detector = None
