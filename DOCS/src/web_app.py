"""
Interactive Web Application & Dashboard for Modern Face Detection System.
Built with Streamlit for intuitive browser-based face analytics, image processing, and camera feeds.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import cv2
import streamlit as st
from PIL import Image

# Ensure root directory in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.core.detector_factory import create_detector
from src.utils.visualizer import Visualizer
from src.utils.snapshot import SnapshotManager
from src.config import VisualizerConfig, SnapshotConfig

# Page configuration
st.set_page_config(
    page_title="AI Face Detection & Tracking System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for rich UI styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00FF80, #00B4DB);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #8E9AAF;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1E222B;
        border-radius: 10px;
        padding: 15px;
        border-left: 4px solid #00FF80;
        box-shadow: 0 4px 6px rgba(0,0,0,0.15);
    }
    .stButton>button {
        background: linear-gradient(90deg, #00c853, #00b0ff);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 6px;
        transition: 0.3s;
    }
</style>
""", unsafe_allow_html=True)


def get_detector(model_name: str, confidence: float):
    return create_detector(model_name, confidence_threshold=confidence)


def main():
    st.markdown('<div class="main-title">⚡ Modern Face Detection & Tracking Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">High-Performance Computer Vision Engine powered by MediaPipe BlazeFace & OpenCV YuNet</div>', unsafe_allow_html=True)

    # Sidebar Settings
    st.sidebar.header("⚙️ Model Configuration")
    model_choice = st.sidebar.selectbox(
        "Detection Engine",
        ["mediapipe", "yunet"],
        format_func=lambda x: "🚀 MediaPipe BlazeFace (Ultra-Fast)" if x == "mediapipe" else "🎯 OpenCV YuNet DNN (High-Precision)",
    )

    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=1.00,
        value=0.60,
        step=0.05,
    )

    st.sidebar.markdown("---")
    st.sidebar.header("🎨 Visual Overlay Options")
    show_bbox = st.sidebar.checkbox("Show Bounding Box", value=True)
    show_landmarks = st.sidebar.checkbox("Show Facial Landmarks", value=True)
    show_osd = st.sidebar.checkbox("Show Telemetry Header", value=True)
    show_conf = st.sidebar.checkbox("Show Confidence Tag", value=True)

    viz_config = VisualizerConfig(
        show_bbox=show_bbox,
        show_landmarks=show_landmarks,
        show_osd=show_osd,
        show_confidence=show_conf,
    )
    visualizer = Visualizer(viz_config)
    snapshot_mgr = SnapshotManager(SnapshotConfig(output_dir="data/snapshots"))

    # Tabs
    tab_image, tab_camera, tab_logs = st.tabs(["🖼️ Image Analysis", "📷 Live Camera Input", "📊 Analytics & Logs"])

    # TAB 1: Static Image Analysis
    with tab_image:
        col_up, col_res = st.columns([1, 2])

        with col_up:
            st.subheader("Upload Image")
            uploaded_file = st.file_uploader(
                "Choose a photo (.jpg, .jpeg, .png)",
                type=["jpg", "jpeg", "png"],
            )

            # Sample buttons
            st.markdown("Or try a sample:")
            use_sample = st.button("Generate & Test Synthetic Sample")

        frame = None
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        elif use_sample:
            # Create synthetic canvas with facial markers
            frame = np.full((600, 800, 3), 40, dtype=np.uint8)
            cv2.circle(frame, (400, 300), 120, (200, 180, 160), -1)  # Face
            cv2.circle(frame, (355, 270), 16, (40, 40, 40), -1)      # Left eye
            cv2.circle(frame, (445, 270), 16, (40, 40, 40), -1)      # Right eye
            cv2.line(frame, (400, 290), (400, 330), (120, 100, 90), 4) # Nose
            cv2.ellipse(frame, (400, 360), (40, 20), 0, 0, 180, (50, 50, 180), 4) # Mouth

        if frame is not None:
            # Run inference
            detector = get_detector(model_choice, conf_threshold)
            t0 = time.perf_counter()
            detections = detector.detect(frame)
            lat_ms = (time.perf_counter() - t0) * 1000.0

            rendered = visualizer.draw(
                frame,
                detections,
                fps=0.0,
                latency_ms=lat_ms,
                model_name=model_choice,
            )

            with col_res:
                st.subheader("Detection Result")
                m1, m2, m3 = st.columns(3)
                m1.metric("Faces Detected", len(detections))
                m2.metric("Inference Latency", f"{lat_ms:.2f} ms")
                m3.metric("Engine", model_choice.upper())

                # Display RGB
                st.image(cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB), use_container_width=True)

            # Face Crops and Metadata
            if detections:
                st.markdown("---")
                st.subheader("🔍 Cropped Face Gallery & Telemetry")
                face_cols = st.columns(min(len(detections), 4))

                records = []
                for i, det in enumerate(detections):
                    det.track_id = i + 1
                    crop = snapshot_mgr.crop_face(frame, det.bbox)
                    with face_cols[i % 4]:
                        if crop is not None and crop.size > 0:
                            st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), caption=f"Face #{i+1} ({det.confidence*100:.1f}%)")
                    
                    records.append({
                        "Face #": i + 1,
                        "Confidence": f"{det.confidence*100:.1f}%",
                        "BBox (x, y, w, h)": f"({det.x}, {det.y}, {det.w}, {det.h})",
                        "Center": f"({det.center[0]}, {det.center[1]})",
                        "Landmarks Count": len(det.landmarks),
                    })

                st.table(pd.DataFrame(records))
            else:
                st.info("No faces detected with current confidence threshold. Try lowering the threshold.")

    # TAB 2: Live Camera Snapshot
    with tab_camera:
        st.subheader("📷 Capture Photo from Webcam")
        cam_photo = st.camera_input("Take a snapshot to run face detection")

        if cam_photo is not None:
            bytes_data = cam_photo.getvalue()
            cam_frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

            detector = get_detector(model_choice, conf_threshold)
            t0 = time.perf_counter()
            cam_detections = detector.detect(cam_frame)
            cam_lat = (time.perf_counter() - t0) * 1000.0

            for i, d in enumerate(cam_detections):
                d.track_id = i + 1

            cam_rendered = visualizer.draw(
                cam_frame,
                cam_detections,
                fps=0.0,
                latency_ms=cam_lat,
                model_name=model_choice,
            )

            c1, c2 = st.columns([2, 1])
            with c1:
                st.image(cv2.cvtColor(cam_rendered, cv2.COLOR_BGR2RGB), caption="Live Processed Frame", use_container_width=True)
            with c2:
                st.metric("Detected Faces", len(cam_detections))
                st.metric("Inference Time", f"{cam_lat:.2f} ms")
                if cam_detections:
                    st.success(f"Successfully processed {len(cam_detections)} face(s)!")

    # TAB 3: Logs and History
    with tab_logs:
        st.subheader("📊 Detection Event Logs")
        log_dir = "data/logs"
        if os.path.exists(log_dir):
            log_files = [f for f in os.listdir(log_dir) if f.endswith(".csv")]
            if log_files:
                selected_log = st.selectbox("Select Log File", log_files)
                df = pd.read_csv(os.path.join(log_dir, selected_log))
                st.dataframe(df, use_container_width=True)

                st.download_button(
                    "📥 Download CSV Log",
                    data=df.to_csv(index=False),
                    file_name=selected_log,
                    mime="text/csv",
                )
            else:
                st.info("No CSV detection logs found yet. Run the CLI / live app to generate real-time logs.")
        else:
            st.info("Log directory is empty.")


if __name__ == "__main__":
    main()
