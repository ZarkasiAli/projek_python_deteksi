"""
Advanced Visualizer and On-Screen Display (OSD) Engine.
Renders futuristic HUD bounding boxes, antialiased facial landmarks, and telemetry dashboards.
"""

from typing import List, Tuple, Optional, Dict
import cv2
import numpy as np

from src.core.base_detector import DetectionResult
from src.config import VisualizerConfig


class Visualizer:
    """
    Renders high-contrast, modern HUD overlays on OpenCV frames.
    """

    def __init__(self, config: Optional[VisualizerConfig] = None) -> None:
        self.config = config or VisualizerConfig()

    @staticmethod
    def _draw_corner_rect(
        img: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int],
        thickness: int = 2,
        corner_len: int = 15,
    ) -> None:
        """Draws a modern HUD box with emphasized corner brackets."""
        x1, y1 = pt1
        x2, y2 = pt2

        # Main rectangle with thin line
        cv2.rectangle(img, (x1, y1), (x2, y2), color, max(1, thickness // 2), cv2.LINE_AA)

        # Ensure corner length does not exceed half the width/height
        c_w = min(corner_len, abs(x2 - x1) // 3)
        c_h = min(corner_len, abs(y2 - y1) // 3)

        # Top-Left
        cv2.line(img, (x1, y1), (x1 + c_w, y1), color, thickness + 1, cv2.LINE_AA)
        cv2.line(img, (x1, y1), (x1, y1 + c_h), color, thickness + 1, cv2.LINE_AA)

        # Top-Right
        cv2.line(img, (x2, y1), (x2 - c_w, y1), color, thickness + 1, cv2.LINE_AA)
        cv2.line(img, (x2, y1), (x2, y1 + c_h), color, thickness + 1, cv2.LINE_AA)

        # Bottom-Left
        cv2.line(img, (x1, y2), (x1 + c_w, y2), color, thickness + 1, cv2.LINE_AA)
        cv2.line(img, (x1, y2), (x1, y2 - c_h), color, thickness + 1, cv2.LINE_AA)

        # Bottom-Right
        cv2.line(img, (x2, y2), (x2 - c_w, y2), color, thickness + 1, cv2.LINE_AA)
        cv2.line(img, (x2, y2), (x2, y2 - c_h), color, thickness + 1, cv2.LINE_AA)

    def draw_osd_header(
        self,
        frame: np.ndarray,
        fps: float,
        latency_ms: float,
        face_count: int,
        model_name: str,
    ) -> None:
        """Renders top telemetry status bar."""
        h, w = frame.shape[:2]
        header_h = 44

        # Semi-transparent dark background banner
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, header_h), (20, 24, 30), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Status accent line
        cv2.line(frame, (0, header_h), (w, header_h), (0, 220, 130), 2, cv2.LINE_AA)

        # Metrics text
        fps_color = (0, 255, 128) if fps >= 25.0 else ((0, 215, 255) if fps >= 15.0 else (0, 70, 255))
        fps_text = f"FPS: {fps:4.1f}"
        lat_text = f"LATENCY: {latency_ms:4.1f}ms"
        faces_text = f"FACES: {face_count}"
        model_text = f"ENGINE: {model_name.upper()}"

        font = cv2.FONT_HERSHEY_DUPLEX
        scale = 0.52
        thick = 1

        # Draw metrics spaced across the header
        cv2.putText(frame, "DETECTION SYSTEM", (15, 28), font, scale + 0.05, (255, 255, 255), thick + 1, cv2.LINE_AA)
        cv2.putText(frame, model_text, (220, 28), font, scale, (200, 200, 200), thick, cv2.LINE_AA)
        cv2.putText(frame, fps_text, (w - 380, 28), font, scale, fps_color, thick + 1, cv2.LINE_AA)
        cv2.putText(frame, lat_text, (w - 240, 28), font, scale, (220, 220, 220), thick, cv2.LINE_AA)
        cv2.putText(frame, faces_text, (w - 90, 28), font, scale, (0, 255, 255), thick + 1, cv2.LINE_AA)

    def draw(
        self,
        frame: np.ndarray,
        detections: List[DetectionResult],
        fps: float = 0.0,
        latency_ms: float = 0.0,
        model_name: str = "MediaPipe",
    ) -> np.ndarray:
        """
        Render all visual detection overlays onto the frame.
        """
        output = frame.copy()
        h, w = output.shape[:2]

        for det in detections:
            x, y, bw, bh = det.bbox
            x2, y2 = x + bw, y + bh

            # Dynamic color based on confidence
            if det.confidence >= 0.8:
                box_color = (0, 255, 128)  # Bright Mint
            elif det.confidence >= 0.6:
                box_color = (0, 220, 255)  # Gold/Yellow
            else:
                box_color = (100, 100, 255)  # Red/Coral

            if self.config.show_bbox:
                # Soft fill inside box
                sub_img = output[max(0, y):min(h, y2), max(0, x):min(w, x2)]
                if sub_img.size > 0:
                    tint = np.full_like(sub_img, box_color, dtype=np.uint8)
                    cv2.addWeighted(tint, 0.08, sub_img, 0.92, 0, sub_img)

                # Corner rectangle HUD
                self._draw_corner_rect(
                    output,
                    (x, y),
                    (x2, y2),
                    box_color,
                    thickness=self.config.line_thickness,
                )

            # Draw Facial Landmarks
            if self.config.show_landmarks and det.landmarks:
                for name, (lx, ly) in det.landmarks.items():
                    if 0 <= lx < w and 0 <= ly < h:
                        # Inner dot
                        cv2.circle(output, (lx, ly), 3, (255, 255, 255), -1, cv2.LINE_AA)
                        # Outer ring
                        cv2.circle(output, (lx, ly), 5, (255, 140, 0), 1, cv2.LINE_AA)

            # Draw Tag / Badge (Track ID & Confidence)
            tag_parts = []
            if self.config.show_track_id and det.track_id is not None:
                tag_parts.append(f"ID #{det.track_id}")
            if self.config.show_confidence:
                tag_parts.append(f"{det.confidence * 100:.1f}%")

            if tag_parts:
                tag_text = " | ".join(tag_parts)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = self.config.font_scale
                (tw, th), baseline = cv2.getTextSize(tag_text, font, font_scale, 1)

                badge_y1 = max(0, y - th - 12)
                badge_y2 = y
                badge_x1 = x
                badge_x2 = min(w, x + tw + 14)

                # Tag background
                cv2.rectangle(output, (badge_x1, badge_y1), (badge_x2, badge_y2), box_color, -1)
                # Tag text
                cv2.putText(
                    output,
                    tag_text,
                    (badge_x1 + 6, badge_y2 - 6),
                    font,
                    font_scale,
                    (15, 15, 15),
                    1,
                    cv2.LINE_AA,
                )

        # Draw Telemetry OSD Header
        if self.config.show_osd:
            self.draw_osd_header(output, fps, latency_ms, len(detections), model_name)

        return output
