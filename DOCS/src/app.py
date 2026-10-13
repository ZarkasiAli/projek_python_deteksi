"""
Main Application Entry Point for Face Detection and Tracking System.
Supports Live Webcam, Video Files, Image Files, and RTSP streams with real-time HUD and keyboard controls.
"""

import argparse
import sys
import os
import time
import cv2
import numpy as np

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.core.detector_factory import create_detector
from src.video.video_stream import ThreadedVideoStream
from src.video.fps_counter import FPSCounter
from src.tracking.tracker import FaceTracker
from src.utils.visualizer import Visualizer
from src.utils.logger import DetectionLogger
from src.utils.snapshot import SnapshotManager
from src.config import AppConfig


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="High-Performance Modern Face Detection & Tracking System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Input source: Webcam index (e.g. 0), video file path (.mp4), image path (.jpg/.png), or RTSP URL",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["mediapipe", "yunet"],
        default="mediapipe",
        help="Face detection model backend",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.6,
        help="Detection confidence threshold (0.1 to 1.0)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Camera capture width",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Camera capture height",
    )
    parser.add_argument(
        "--no-track",
        action="store_true",
        help="Disable face tracking and ID assignment",
    )
    parser.add_argument(
        "--no-flip",
        action="store_true",
        help="Do not flip camera horizontally (mirror mode)",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable CSV/JSON event logging",
    )
    parser.add_argument(
        "--no-snapshot",
        action="store_true",
        help="Disable snapshot manager",
    )
    return parser.parse_args()


def process_image(image_path: str, model_type: str, conf: float) -> None:
    """Process and display a single static image."""
    if not os.path.exists(image_path):
        print(f"[Error] Image file not found: {image_path}")
        return

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[Error] Failed to decode image: {image_path}")
        return

    print(f"[Info] Running {model_type.upper()} face detector on {image_path}...")
    detector = create_detector(model_type, confidence_threshold=conf)
    visualizer = Visualizer()
    snapshot_mgr = SnapshotManager()

    start_t = time.perf_counter()
    detections = detector.detect(frame)
    latency_ms = (time.perf_counter() - start_t) * 1000.0

    print(f"[Result] Detected {len(detections)} face(s) in {latency_ms:.2f}ms")

    rendered = visualizer.draw(
        frame,
        detections,
        fps=0.0,
        latency_ms=latency_ms,
        model_name=model_type,
    )

    # Save output image
    out_path = os.path.splitext(image_path)[0] + "_detected.jpg"
    cv2.imwrite(out_path, rendered)
    print(f"[Info] Saved detection visual output to {out_path}")

    # Auto-save cropped faces
    for i, det in enumerate(detections):
        det.track_id = i + 1
        crop_path = snapshot_mgr.save_face_snapshot(frame, det, prefix="image_face")
        if crop_path:
            print(f"  - Face #{i+1}: Conf={det.confidence*100:.1f}%, Saved crop: {crop_path}")

    window_name = f"Face Detection - {model_type.upper()}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, rendered)
    print("\n[Controls] Press any key to close the image window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    detector.close()


def run_stream(args: argparse.Namespace) -> None:
    """Run real-time video stream processing loop."""
    config = AppConfig()
    config.ensure_directories()
    config.detector.model_type = args.model
    config.detector.confidence_threshold = args.conf
    config.tracking.enabled = not args.no_track
    config.logging.enabled = not args.no_log
    config.snapshot.enabled = not args.no_snapshot
    config.stream.flip_horizontal = not args.no_flip

    # Parse source (int for webcam, str for file/rtsp)
    src: any = int(args.source) if args.source.isdigit() else args.source

    print("=" * 65)
    print("  🚀 FACE DETECTION & TRACKING SYSTEM (MODERN AI PIPELINE)")
    print("=" * 65)
    print(f" • Input Source     : {src}")
    print(f" • Active Detector  : {config.detector.model_type.upper()}")
    print(f" • Confidence Limit : {config.detector.confidence_threshold:.2f}")
    print(f" • Tracking         : {'ENABLED' if config.tracking.enabled else 'DISABLED'}")
    print(f" • Event Logging    : {'ENABLED' if config.logging.enabled else 'DISABLED'}")
    print("-" * 65)
    print(" Keyboard Controls:")
    print("  [q / ESC]  Quit application")
    print("  [m]        Switch Detector Engine (MediaPipe <-> YuNet)")
    print("  [t]        Toggle Face Tracking (Unique IDs)")
    print("  [s]        Take Snapshot (Full frame & Cropped faces)")
    print("  [l]        Toggle Facial Landmarks")
    print("  [b]        Toggle Bounding Boxes")
    print("  [o]        Toggle OSD Telemetry Header")
    print("  [+] / [-]  Increase / Decrease Confidence Threshold")
    print("=" * 65)

    # Initialize modules
    active_model = config.detector.model_type
    detector = create_detector(active_model, confidence_threshold=config.detector.confidence_threshold)
    tracker = FaceTracker() if config.tracking.enabled else None
    visualizer = Visualizer(config.visualizer)
    logger = DetectionLogger(config.logging) if config.logging.enabled else None
    snapshot_mgr = SnapshotManager(config.snapshot) if config.snapshot.enabled else None
    fps_counter = FPSCounter(window_size=30)

    # Initialize Threaded Video Stream
    print(f"[Info] Initializing video stream from source '{src}'...")
    stream = ThreadedVideoStream(
        src=src,
        width=args.width,
        height=args.height,
        target_fps=config.stream.target_fps,
    ).start()

    if not stream.is_opened:
        print(f"[Error] Could not open video source: {src}")
        stream.stop()
        return

    window_name = "Modern Face Detection System"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_idx = 0
    running = True

    try:
        while running:
            fps_counter.tick()

            grabbed, frame = stream.read()
            if not grabbed or frame is None:
                # End of video file or stream disconnect
                time.sleep(0.01)
                if isinstance(src, str) and os.path.exists(src):
                    print("[Info] Video stream reached end of file.")
                    break
                continue

            frame_idx += 1

            if config.stream.flip_horizontal and isinstance(src, int):
                frame = cv2.flip(frame, 1)

            # AI Inference
            fps_counter.start_inference()
            detections = detector.detect(frame)
            fps_counter.end_inference()

            # Tracking
            if config.tracking.enabled and tracker is not None:
                detections = tracker.update(detections)

            # Snapshot & Logging
            saved_snapshots = {}
            if snapshot_mgr is not None and config.snapshot.enabled:
                saved_snapshots = snapshot_mgr.auto_snapshot_faces(frame, detections)

            if logger is not None:
                logger.log_frame_detections(
                    frame_idx=frame_idx,
                    detections=detections,
                    model_name=active_model,
                    snapshots=saved_snapshots,
                )

            # Visual Rendering
            rendered = visualizer.draw(
                frame,
                detections,
                fps=fps_counter.fps,
                latency_ms=fps_counter.latency_ms,
                model_name=active_model,
            )

            cv2.imshow(window_name, rendered)

            # Keyboard Events
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                print("\n[Info] Exit key pressed. Terminating...")
                running = False
            elif key in (ord('m'), ord('M')):
                # Toggle detector model
                active_model = "yunet" if active_model == "mediapipe" else "mediapipe"
                detector.close()
                detector = create_detector(active_model, confidence_threshold=config.detector.confidence_threshold)
                print(f"[Action] Switched detector engine to: {active_model.upper()}")
            elif key in (ord('t'), ord('T')):
                config.tracking.enabled = not config.tracking.enabled
                if config.tracking.enabled and tracker is None:
                    tracker = FaceTracker()
                print(f"[Action] Tracking: {'ENABLED' if config.tracking.enabled else 'DISABLED'}")
            elif key in (ord('s'), ord('S')):
                # Manual snapshot
                if snapshot_mgr is not None:
                    full_snap = snapshot_mgr.save_full_frame(frame, prefix="manual_frame")
                    print(f"[Snapshot] Saved full frame: {full_snap}")
                    for det in detections:
                        face_snap = snapshot_mgr.save_face_snapshot(frame, det, prefix="manual_face")
                        print(f"  - Saved face crop: {face_snap}")
            elif key in (ord('l'), ord('L')):
                config.visualizer.show_landmarks = not config.visualizer.show_landmarks
                print(f"[Action] Landmarks: {'SHOWN' if config.visualizer.show_landmarks else 'HIDDEN'}")
            elif key in (ord('b'), ord('B')):
                config.visualizer.show_bbox = not config.visualizer.show_bbox
                print(f"[Action] Bounding Box: {'SHOWN' if config.visualizer.show_bbox else 'HIDDEN'}")
            elif key in (ord('o'), ord('O')):
                config.visualizer.show_osd = not config.visualizer.show_osd
                print(f"[Action] OSD Header: {'SHOWN' if config.visualizer.show_osd else 'HIDDEN'}")
            elif key in (ord('+'), ord('=')):
                new_conf = min(0.95, detector.confidence_threshold + 0.05)
                detector.set_confidence_threshold(new_conf)
                config.detector.confidence_threshold = new_conf
                print(f"[Action] Confidence threshold increased to: {new_conf:.2f}")
            elif key in (ord('-'), ord('_')):
                new_conf = max(0.10, detector.confidence_threshold - 0.05)
                detector.set_confidence_threshold(new_conf)
                config.detector.confidence_threshold = new_conf
                print(f"[Action] Confidence threshold decreased to: {new_conf:.2f}")

    finally:
        print("[Info] Cleaning up resources...")
        stream.stop()
        cv2.destroyAllWindows()
        detector.close()
        if logger is not None:
            logger.close()
            print(f"[Info] Event logs saved to: {logger.csv_path}")
        print("[Info] Application closed safely.")


def main() -> None:
    args = parse_arguments()
    src = args.source

    # Check if input is a static image file
    if isinstance(src, str) and any(src.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]):
        process_image(src, args.model, args.conf)
    else:
        run_stream(args)


if __name__ == "__main__":
    main()
