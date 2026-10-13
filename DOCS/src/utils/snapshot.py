"""
Snapshot and Face Cropping Manager.
Saves high-resolution face crops and full-frame snapshots with safe padding.
"""

import os
import time
from typing import Optional, Dict, Tuple, List
import cv2
import numpy as np

from src.core.base_detector import DetectionResult
from src.config import SnapshotConfig


class SnapshotManager:
    """
    Handles capturing and exporting detected faces and frames.
    """

    def __init__(self, config: Optional[SnapshotConfig] = None) -> None:
        self.config = config or SnapshotConfig()
        os.makedirs(self.config.output_dir, exist_ok=True)
        self._last_snapshot_time: Dict[int, float] = {}  # track_id -> timestamp

    def crop_face(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        margin_ratio: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        """
        Crops a face from the frame with a proportional margin.
        """
        if frame is None or frame.size == 0:
            return None

        h, w = frame.shape[:2]
        x, y, bw, bh = bbox
        margin = margin_ratio if margin_ratio is not None else self.config.margin_ratio

        pad_x = int(bw * margin)
        pad_y = int(bh * margin)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + bw + pad_x)
        y2 = min(h, y + bh + pad_y)

        if x2 <= x1 or y2 <= y1:
            return None

        return frame[y1:y2, x1:x2].copy()

    def save_face_snapshot(
        self,
        frame: np.ndarray,
        detection: DetectionResult,
        prefix: str = "face",
    ) -> Optional[str]:
        """
        Crop and save face image to output directory.
        
        Returns:
            Relative file path of saved snapshot or None if failed.
        """
        if not self.config.enabled:
            return None

        crop = self.crop_face(frame, detection.bbox)
        if crop is None or crop.size == 0:
            return None

        ts = int(time.time() * 1000)
        track_str = f"id_{detection.track_id}" if detection.track_id is not None else "det"
        filename = f"{prefix}_{track_str}_{ts}.jpg"
        filepath = os.path.join(self.config.output_dir, filename)

        success = cv2.imwrite(filepath, crop)
        return filepath if success else None

    def save_full_frame(self, frame: np.ndarray, prefix: str = "snapshot") -> Optional[str]:
        """
        Save the entire frame snapshot.
        """
        if frame is None or frame.size == 0:
            return None

        ts = int(time.time() * 1000)
        filename = f"{prefix}_{ts}.jpg"
        filepath = os.path.join(self.config.output_dir, filename)

        success = cv2.imwrite(filepath, frame)
        return filepath if success else None

    def auto_snapshot_faces(
        self,
        frame: np.ndarray,
        detections: List[DetectionResult],
        cooldown_sec: float = 5.0,
    ) -> Dict[int, str]:
        """
        Automatically save snapshots of newly detected or qualified faces with cooldown.
        
        Returns:
            Dictionary mapping track_id to saved file path.
        """
        saved: Dict[int, str] = {}
        now = time.time()

        for det in detections:
            tid = det.track_id
            if tid is None or det.confidence < self.config.min_confidence:
                continue

            last_time = self._last_snapshot_time.get(tid, 0.0)
            if now - last_time >= cooldown_sec:
                path = self.save_face_snapshot(frame, det)
                if path:
                    saved[tid] = path
                    self._last_snapshot_time[tid] = now

        return saved
