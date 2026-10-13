"""
Multi-Face Object Tracker with IoU and Centroid Association.
Assigns unique, persistent tracking IDs to faces across consecutive video frames.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from src.core.base_detector import DetectionResult


class TrackedFace:
    """Represents an active or lost tracked face state."""

    def __init__(self, track_id: int, result: DetectionResult) -> None:
        self.track_id = track_id
        self.bbox = result.bbox  # (x, y, w, h)
        self.confidence = result.confidence
        self.landmarks = result.landmarks
        self.center = result.center
        self.disappeared_count = 0
        self.total_frames_tracked = 1
        self.history: List[Tuple[int, int]] = [result.center]

    def update(self, result: DetectionResult) -> None:
        """Update tracker state with new matched detection."""
        self.bbox = result.bbox
        self.confidence = result.confidence
        self.landmarks = result.landmarks
        self.center = result.center
        self.disappeared_count = 0
        self.total_frames_tracked += 1
        self.history.append(result.center)
        if len(self.history) > 30:
            self.history.pop(0)

    def mark_missed(self) -> None:
        """Increment disappearance counter when face is not matched in a frame."""
        self.disappeared_count += 1


class FaceTracker:
    """
    Online face tracking system using IoU (Intersection over Union) and Euclidean distance.
    """

    def __init__(
        self,
        max_disappeared: int = 15,
        min_iou_threshold: float = 0.3,
        max_distance_threshold: float = 80.0,
    ) -> None:
        self.max_disappeared = max_disappeared
        self.min_iou_threshold = min_iou_threshold
        self.max_distance_threshold = max_distance_threshold

        self._next_track_id: int = 1
        self.tracked_objects: Dict[int, TrackedFace] = {}

    @staticmethod
    def compute_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
        """
        Compute Intersection over Union between two bounding boxes (x, y, w, h).
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

        inter_w = max(0, xB - xA)
        inter_h = max(0, yB - yA)
        inter_area = inter_w * inter_h

        boxA_area = boxA[2] * boxA[3]
        boxB_area = boxB[2] * boxB[3]
        union_area = boxA_area + boxB_area - inter_area

        if union_area <= 0:
            return 0.0
        return float(inter_area / union_area)

    @staticmethod
    def compute_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """Compute Euclidean distance between two center points."""
        return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))

    def register(self, detection: DetectionResult) -> int:
        """Register a new tracked face with a unique ID."""
        track_id = self._next_track_id
        self._next_track_id += 1
        self.tracked_objects[track_id] = TrackedFace(track_id, detection)
        detection.track_id = track_id
        return track_id

    def deregister(self, track_id: int) -> None:
        """Remove tracked face after exceeding max disappearance frames."""
        if track_id in self.tracked_objects:
            del self.tracked_objects[track_id]

    def update(self, detections: List[DetectionResult]) -> List[DetectionResult]:
        """
        Update tracked objects given new detection results.
        
        Args:
            detections: List of DetectionResult from current frame.

        Returns:
            Updated List of DetectionResult with `track_id` assigned.
        """
        if len(detections) == 0:
            # Mark all active tracks as missed
            for track_id in list(self.tracked_objects.keys()):
                self.tracked_objects[track_id].mark_missed()
                if self.tracked_objects[track_id].disappeared_count > self.max_disappeared:
                    self.deregister(track_id)
            return detections

        if len(self.tracked_objects) == 0:
            # Register all current detections as new tracks
            for det in detections:
                self.register(det)
            return detections

        # Match active tracks with new detections using IoU and Distance Cost Matrix
        track_ids = list(self.tracked_objects.keys())
        cost_matrix = np.zeros((len(track_ids), len(detections)), dtype=np.float32)

        for i, tid in enumerate(track_ids):
            tracked_face = self.tracked_objects[tid]
            for j, det in enumerate(detections):
                iou = self.compute_iou(tracked_face.bbox, det.bbox)
                dist = self.compute_distance(tracked_face.center, det.center)
                
                # Higher score is better match (combination of IoU and proximity)
                if iou >= self.min_iou_threshold:
                    match_score = iou * 2.0
                elif dist <= self.max_distance_threshold:
                    match_score = 1.0 - (dist / self.max_distance_threshold)
                else:
                    match_score = 0.0

                cost_matrix[i, j] = match_score

        # Greedy matching from highest match score
        used_tracks = set()
        used_detections = set()

        # Sort all possible matches in descending order of score
        matches = []
        for i in range(len(track_ids)):
            for j in range(len(detections)):
                if cost_matrix[i, j] > 0.0:
                    matches.append((cost_matrix[i, j], i, j))

        matches.sort(key=lambda x: x[0], reverse=True)

        for score, i, j in matches:
            if i in used_tracks or j in used_detections:
                continue

            tid = track_ids[i]
            self.tracked_objects[tid].update(detections[j])
            detections[j].track_id = tid

            used_tracks.add(i)
            used_detections.add(j)

        # Handle unmatched active tracks
        for i, tid in enumerate(track_ids):
            if i not in used_tracks:
                self.tracked_objects[tid].mark_missed()
                if self.tracked_objects[tid].disappeared_count > self.max_disappeared:
                    self.deregister(tid)

        # Handle unmatched new detections (register new track)
        for j, det in enumerate(detections):
            if j not in used_detections:
                self.register(det)

        return detections

    def reset(self) -> None:
        """Reset all tracking states and counters."""
        self._next_track_id = 1
        self.tracked_objects.clear()
