"""
Detection Event Logger supporting CSV and JSON/JSONL output formats.
"""

import csv
import json
import os
import time
from typing import List, Optional
from src.core.base_detector import DetectionResult
from src.config import LoggingConfig


class DetectionLogger:
    """
    Records detection events and track telemetry to persistent log files.
    """

    def __init__(self, config: Optional[LoggingConfig] = None) -> None:
        self.config = config or LoggingConfig()
        os.makedirs(self.config.log_dir, exist_ok=True)

        session_id = time.strftime("%Y%m%d_%H%M%S")
        self.jsonl_path = os.path.join(self.config.log_dir, f"detections_{session_id}.jsonl")
        self.csv_path = os.path.join(self.config.log_dir, f"detections_{session_id}.csv")

        self._csv_file = None
        self._csv_writer = None
        self._jsonl_file = None

        if self.config.enabled:
            self._init_files()

    def _init_files(self) -> None:
        if self.config.export_format in ("csv", "both"):
            self._csv_file = open(self.csv_path, mode="w", newline="", encoding="utf-8")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow([
                "timestamp",
                "frame_idx",
                "track_id",
                "confidence",
                "x",
                "y",
                "w",
                "h",
                "model",
                "snapshot",
            ])

        if self.config.export_format in ("json", "jsonl", "both"):
            self._jsonl_file = open(self.jsonl_path, mode="w", encoding="utf-8")

    def log_detection(
        self,
        frame_idx: int,
        detection: DetectionResult,
        model_name: str = "MediaPipe",
        snapshot_path: Optional[str] = None,
    ) -> None:
        """Log a single detection record."""
        if not self.config.enabled:
            return

        ts = time.time()
        iso_time = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts)) + f".{int(ts % 1 * 1000):03d}"

        # Write to CSV
        if self._csv_writer is not None:
            self._csv_writer.writerow([
                iso_time,
                frame_idx,
                detection.track_id if detection.track_id is not None else "",
                f"{detection.confidence:.4f}",
                detection.x,
                detection.y,
                detection.w,
                detection.h,
                model_name,
                snapshot_path or "",
            ])
            self._csv_file.flush()

        # Write to JSONL
        if self._jsonl_file is not None:
            record = {
                "timestamp": iso_time,
                "frame_idx": frame_idx,
                "track_id": detection.track_id,
                "confidence": round(detection.confidence, 4),
                "bbox": {"x": detection.x, "y": detection.y, "w": detection.w, "h": detection.h},
                "landmarks": detection.landmarks,
                "model": model_name,
                "snapshot": snapshot_path,
            }
            self._jsonl_file.write(json.dumps(record) + "\n")
            self._jsonl_file.flush()

    def log_frame_detections(
        self,
        frame_idx: int,
        detections: List[DetectionResult],
        model_name: str = "MediaPipe",
        snapshots: Optional[dict] = None,
    ) -> None:
        """Log all detections in a frame."""
        if not self.config.enabled:
            return
        snapshots = snapshots or {}
        for det in detections:
            snap = snapshots.get(det.track_id)
            self.log_detection(frame_idx, det, model_name=model_name, snapshot_path=snap)

    def close(self) -> None:
        """Close opened log file descriptors."""
        if self._csv_file is not None and not self._csv_file.closed:
            self._csv_file.close()
        if self._jsonl_file is not None and not self._jsonl_file.closed:
            self._jsonl_file.close()

    def __enter__(self) -> "DetectionLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
