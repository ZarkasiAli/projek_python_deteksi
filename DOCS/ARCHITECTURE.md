# Arsitektur Sistem Deteksi Wajah Berbasis Python (System Architecture)

Dokumen ini menjelaskan arsitektur perangkat lunak, aliran data (*data pipeline*), pola desain, dan struktur modular dari **Sistem Deteksi & Pelacakan Wajah Modern**.

---

## 1. Diagram Tingkat Tinggi (High-Level Architecture)

```
                       ┌─────────────────────────────────────────┐
                       │           Multi-Source Input            │
                       │ (Webcam / Video File / Image / RTSP)    │
                       └───────────────────┬─────────────────────┘
                                           │
                                           ▼
                       ┌─────────────────────────────────────────┐
                       │     ThreadedVideoStream (Producer)      │
                       │  - Asynchronous frame grabbing          │
                       │  - Non-blocking lock-free frame buffer  │
                       │  - Zero latency drift & frame-skip      │
                       └───────────────────┬─────────────────────┘
                                           │
                                           ▼
                       ┌─────────────────────────────────────────┐
                       │     Core Detector Engine (Consumer)     │
                       │   ┌─────────────────────────────────┐   │
                       │   │          BaseDetector           │   │
                       │   │     (Abstract Base Class)       │   │
                       │   └───────▲─────────────────▲───────┘   │
                       │           │                 │           │
                       │   ┌───────┴──────┐   ┌──────┴───────┐   │
                       │   │  MediaPipe   │   │ OpenCV YuNet │   │
                       │   │  (BlazeFace) │   │  (DNN ONNX)  │   │
                       │   └──────────────┘   └──────────────┘   │
                       └───────────────────┬─────────────────────┘
                                           │  List[DetectionResult]
                                           ▼
                       ┌─────────────────────────────────────────┐
                       │         Face Tracker (Tracking)         │
                       │  - IoU & Centroid Matching Matrix       │
                       │  - Persistent Face ID Assignment        │
                       │  - Lifetime & Disappearance Handling    │
                       └───────────────────┬─────────────────────┘
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    │                      │                      │
                    ▼                      ▼                      ▼
        ┌───────────────────────┐ ┌───────────────────┐ ┌───────────────────┐
        │       Visualizer      │ │ DetectionLogger   │ │  SnapshotManager  │
        │ - HUD Bounding Box    │ │ - JSONL Events    │ │ - Face Cropping   │
        │ - Antialiased Points  │ │ - CSV Export      │ │ - Full Frame Save │
        │ - Telemetry OSD Bar   │ │ - Timestamp Logs  │ │ - Auto-Capture    │
        └───────────┬───────────┘ └───────────────────┘ └───────────────────┘
                    │
                    ▼
        ┌─────────────────────────────────────────┐
        │             User Interfaces             │
        │  1. Real-Time OpenCV OSD Window (CLI)   │
        │  2. Interactive Streamlit Web Studio    │
        └─────────────────────────────────────────┘
```

---

## 2. Rincian Modul & Komponen

### 2.1 Core Detection Engine (`src/core/`)
* **`BaseDetector` (Abstract Base Class):** Mendefinisikan kontrak interface yang seragam (`detect(frame) -> List[DetectionResult]`, `set_confidence_threshold()`, `close()`).
* **`DetectionResult` (Dataclass):** Struktur data standar berisi koordinat bounding box `(x, y, w, h)`, `confidence`, titik landmark wajah, serta `track_id`.
* **`MediaPipeDetector`:** Menggunakan arsitektur model BlazeFace dari MediaPipe Tasks Vision. Sangat optimal untuk CPU (latensi ~10-15ms) dengan 6 landmark wajah.
* **`YuNetDetector`:** Menggunakan OpenCV DNN Face Detector (YuNet ONNX). Menghasilkan deteksi wajah yang akurat pada sudut rotasi ekstrem dan pencahayaan beragam.
* **`create_detector()`:** Factory pattern untuk menginisialisasi detektor berdasarkan konfigurasi dinamis.

### 2.2 Streaming Pipeline (`src/video/`)
* **`ThreadedVideoStream`:** Mengimplementasikan pola **Producer-Consumer**. Thread latar belakang secara kontinu membaca frame kamera ke buffer memori bersama (*shared buffer*), mencegah I/O blocking pada loop inferensi AI.
* **`FPSCounter`:** Menggunakan jendela waktu geser (*moving-average time window*) untuk menghitung FPS riil dan latensi inferensi dalam milidetik (ms).

### 2.3 Pelacakan Objek (`src/tracking/`)
* **`FaceTracker`:** Algoritma pelacakan multi-target berbasis kombinasi *Intersection over Union (IoU)* dan *Euclidean Centroid Distance*. Memberikan ID persisten pada wajah sehingga mencegah *double-counting* saat subjek bergerak di depan kamera.

### 2.4 Utilitas & Penyimpanan (`src/utils/`)
* **`Visualizer`:** Penggambaran HUD modern dengan sudut *corner brackets*, transparansi lembut, visualisasi titik landmark, serta status bar OSD (telemetri FPS, latensi, dan model aktif).
* **`DetectionLogger`:** Pencatatan otomatis setiap deteksi ke format terstruktur (CSV dan JSONLines) untuk kebutuhan audit dan analitik data.
* **`SnapshotManager`:** Pemotongan crop wajah (*margin padding 20%*) dan penyimpanan gambar resolusi tinggi tanpa distorsi.

---

## 3. Format Data Logging

### Contoh Data JSONLines (`data/logs/detections_YYYYMMDD_HHMMSS.jsonl`):
```json
{
  "timestamp": "2026-08-26T14:35:00.123",
  "frame_idx": 45,
  "track_id": 1,
  "confidence": 0.9421,
  "bbox": {"x": 320, "y": 180, "w": 140, "h": 170},
  "landmarks": {
    "right_eye": [360, 220],
    "left_eye": [420, 220],
    "nose_tip": [390, 260],
    "mouth_right": [370, 300],
    "mouth_left": [410, 300]
  },
  "model": "MediaPipe",
  "snapshot": "data/snapshots/face_id_1_1724657700123.jpg"
}
```

---

## 4. Panduan Ekstensi Masa Depan (Extensibility Guide)

1. **Menambahkan Face Recognition (Pengenalan Wajah):**
   - Buat kelas `FaceRecognizer` di `src/recognition/`.
   - Gunakan crop wajah dari `SnapshotManager.crop_face()` untuk diekstrak fiturnya (*feature embedding 512-d*) menggunakan model ArcFace atau FaceNet.
2. **Menambahkan Liveness Detection (Anti-Spoofing):**
   - Buat kelas `LivenessDetector` di `src/liveness/`.
   - Colokkan hasil evaluasi liveness ke dalam `DetectionResult.extra['is_real']`.
