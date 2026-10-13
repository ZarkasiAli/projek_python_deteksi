"""
Centralized Configuration Module for Face Detection and Tracking System.
"""

from dataclasses import dataclass, field
from typing import Optional, Union, Tuple
import os


@dataclass
class DetectorConfig:
    model_type: str = "mediapipe"  # "mediapipe" or "yunet"
    confidence_threshold: float = 0.6
    nms_threshold: float = 0.3
    model_path: Optional[str] = None


@dataclass
class StreamConfig:
    source: Union[int, str] = 0
    width: int = 1280
    height: int = 720
    target_fps: int = 30
    flip_horizontal: bool = True


@dataclass
class TrackingConfig:
    enabled: bool = True
    max_disappeared: int = 15
    min_iou_threshold: float = 0.3
    max_distance_threshold: float = 80.0


@dataclass
class VisualizerConfig:
    show_bbox: bool = True
    show_landmarks: bool = True
    show_track_id: bool = True
    show_confidence: bool = True
    show_osd: bool = True
    theme_color: Tuple[int, int, int] = (0, 255, 128)  # Neon green (BGR)
    accent_color: Tuple[int, int, int] = (255, 128, 0)  # Cyan/Blue (BGR)
    font_scale: float = 0.55
    line_thickness: int = 2


@dataclass
class LoggingConfig:
    enabled: bool = True
    log_dir: str = "data/logs"
    export_format: str = "json"  # "json" or "csv"
    log_snapshots: bool = True


@dataclass
class SnapshotConfig:
    enabled: bool = True
    output_dir: str = "data/snapshots"
    margin_ratio: float = 0.2  # 20% margin around cropped face
    min_confidence: float = 0.65


@dataclass
class AppConfig:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    stream: StreamConfig = field(default_factory=StreamConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    visualizer: VisualizerConfig = field(default_factory=VisualizerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    snapshot: SnapshotConfig = field(default_factory=SnapshotConfig)

    def ensure_directories(self) -> None:
        """Create necessary storage directories if they do not exist."""
        os.makedirs(self.logging.log_dir, exist_ok=True)
        os.makedirs(self.snapshot.output_dir, exist_ok=True)
        os.makedirs("data/models", exist_ok=True)
        os.makedirs("data/samples", exist_ok=True)
