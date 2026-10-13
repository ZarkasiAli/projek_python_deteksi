# 📋 Catatan Fase Pengembangan & Roadmap Sistem Deteksi Wajah

Dokumen ini mencatat seluruh **status pengerjaan per fase**, rincian fitur yang telah diselesaikan, artefak kode yang dihasilkan, serta roadmap pengembangan lanjutan untuk **Sistem Deteksi & Pelacakan Wajah Modern**.

---

## 📌 Ringkasan Status Proyek (Project Progress Overview)

* **Status Keseluruhan:** **SELESAI (100%)** — 4 dari 4 Fase Inti Selesai & Teruji
* **Status Pengujian:** 13 Unit Tests Lolos (**100% Pass**)
* **Bahasa & Ekosistem:** Python 3.10+ / Python 3.13, OpenCV 5.0+, MediaPipe 1.0+, Streamlit 1.62+, Pytest

```
[████████████████████████████████████████] 100% (4/4 Fase Selesai)
```

| Fase | Nama Fase | Status | Tanggal / Versi |
| :--- | :--- | :---: | :---: |
| **Fase 1** | **Fondasi Model & Detektor Inti** | ✅ **Selesai** | v1.0.0 |
| **Fase 2** | **Optimasi Kinerja & Video Pipeline** | ✅ **Selesai** | v1.0.0 |
| **Fase 3** | **Pelacakan Wajah, Snapshot & Logging** | ✅ **Selesai** | v1.0.0 |
| **Fase 4** | **Antarmuka Aplikasi, Dashboard, Tests & Docs** | ✅ **Selesai** | v1.0.0 |

---

## 🔍 Rincian Pengerjaan per Fase

---

### 🟢 Fase 1: Fondasi Model & Detektor Inti (Core Detection Engine)
**Sasaran:** Membangun fondasi detektor berbasis prinsip OOP & SOLID yang modular, menggantikan Haar Cascade dengan model modern (MediaPipe BlazeFace & OpenCV YuNet).

#### ✅ Checklist Pengerjaan:
- [x] Konfigurasi file dependensi [`requirements.txt`](file:///c:/Users/User/Documents/PROJEK_PYTHON_DETEKSI/DOCS/requirements.txt).
- [x] Implementasi Abstract Base Class [`BaseDetector`](file:///c:/Users/User/Documents/PROJEK_PYTHON_DETEKSI/DOCS/src/core/base_detector.py) dan dataclass [`DetectionResult`](file:///c:/Users/User/Documents/PROJEK_PYTHON_DETEKSI/DOCS/src/core/base_detector.py).
- [x] Implementasi [`MediaPipeDetector`](file:///c:/Users/User/Documents/PROJEK_PYTHON_DETEKSI/DOCS/src/core/mediapipe_detector.py) (BlazeFace CPU-optimized, latensi < 15ms, 6 landmarks).
- [x] Implementasi [`YuNetDetector`](file:///c:/Users/User/Documents/PROJEK_PYTHON_DETEKSI/DOCS/src/core/yunet_detector.py) (OpenCV DNN ONNX, rotasi ekstrem, 5 landmarks).
- [x] Pengunduhan & integrasi bobot model otomatis (`blaze_face_short_range.tflite` & `face_detection_yunet_2023mar.onnx`).
- [x] Pembuatan [`detector_factory.py`](file:///c:/Users/User/Documents/PROJEK_PYTHON_DETEKSI/DOCS/src/core/detector_factory.py) untuk dynamic instantiation.
- [x] Unit test detektor di [`tests/test_detector.py`](file:///c:/Users/User/Documents/PROJEK_PYTHON_DETEKSI/DOCS/tests/test_detector.py) (5 tests lulus).

#### 📁 File yang Dihasilkan:
* [`src/core/base_detector.py`](file:///c:/Users/User/Documents/PROJEK_PYTHON_DETEKSI/DOCS/src/core/base_detector.py)
* [`src/core/mediapipe_detector.py`](file:///c:/Users/User/Documents/PROJEK_PYTHON_DETEKSI/DOCS/src/core/mediapipe_detector.py)
* [`src/core/yunet_detector.py`](file:///c:/Users/User/Documents/PROJEK_PYTHON_DETEKSI/DOCS/src/core/yunet_detector.py)
* [`src/core/detector_factory.py`](file:///c:/Users/User/Documents/PROJEK_PYTHON_DETEKSI/DOCS/src/core/detector_factory.py)
* [`src/core/__init__.py`](file:///c:/Users/User/Documents/PROJEK_PYTHON_DETEKSI/DOCS/src/core/__init__.py)
* [`data/models/face_detection_yunet_2023mar.onnx`](file:///c:/Users/User/Documents/PROJEK_PYTHON_DETEKSI/DOCS/data/models/face_detection_yunet_2023mar.onnx)
* [`data/models/blaze_face_short_range.tflite`](file:///c:/Users/User/Documents/PROJEK_PYTHON_DETEKSI/DOCS/data/models/blaze_face_short_range.tflite)

---

### 🟢 Fase 2: Optimasi Kinerja & Video Pipeline
**Sasaran:** Menghilangkan latency spike dan frame drop dengan memisahkan thread decoding kamera dari thread kalkulasi inferensi AI (Producer-Consumer architecture).

#### ✅ Checklist Pengerjaan:
- [x] Implementasi [`ThreadedVideoStream`](file:///c:/Users/User/Documents/PROJEK_PYTHON_DETEKSI/DOCS/src/video/video_stream.py) dengan background daemon thread & buffer lock non-blocking.
- [x] Mekanisme auto-skip frame lama untuk memastikan stream selalu menampilkan frame *real-time*.
- [x] Implementasi [`FPSCounter`](file:///c:/Users/User/Documents/PROJEK_PYTHON_DETEKSI/DOCS/src/video/fps_counter.py) menggunakan *moving-average window* untuk FPS dan latensi inferensi milidetik (ms).
- [x] Implementasi [`AppConfig`](file:///c:/Users/User/Documents/PROJEK_PYTHON_DETEKSI/DOCS/src/config.py) untuk konfigurasi terpusat (resolusi, threshold, auto-flip kamera).
- [x] Unit test pipeline video di [`tests/test_stream.py`](file:///c:/Users/User/Documents/PROJEK_PYTHON_DETEKSI/DOCS/tests/test_stream.py) (3 tests lulus).

#### 📁 File yang Dihasilkan:
* [`src/video/video_stream.py`](file:///c:/Users/User/Documents/PROJEK_PYTHON_DETEKSI/DOCS/src/video/video_stream.py)
* [`src/video/fps_counter.py`](file:///c:/Users/User/Documents/PROJEK_PYTHON_DETEKSI/DOCS/src/video/fps_counter.py)
* [`src/video/__init__.py`](file:///c:/Users/User/Documents/PROJEK_PYTHON_DETEKSI/DOCS/src/video/__init__.py)
* [`src/config.py`](file:///c:/Users/User/Documents/PROJEK_PYTHON_DETEKSI/DOCS/src/config.py)

---

### 🟢 Fase 3: Pelacakan Wajah (Tracking), Snapshot & Logging
**Sasaran:** Memberikan ID unik persisten pada setiap wajah antar-frame, menyediakan fitur crop wajah kualitas tinggi, dan mencatat event deteksi ke format file terstruktur.

#### ✅ Checklist Pengerjaan:
- [x] Implementasi [`FaceTracker`](file:///c:/Users/User/Documents/PROJEK_PYTHON_DETEKSI/DOCS/src/tracking/tracker.py) (IoU + Euclidean Distance cost matrix matching).
- [x] Manajemen track lifetime & auto-deregistration untuk wajah yang keluar dari frame.
- [x] Implementasi [`Visualizer`](file:///c:/Users/User/Documents/PROJEK_PYTHON_DETEKSI/DOCS/src/utils/visualizer.py) (Futuristic HUD corner brackets, landmark dots, dynamic confidence colors, dan OSD header).
- [x] Implementasi [`SnapshotManager`](file:///c:/Users/User/Documents/PROJEK_PYTHON_DETEKSI/DOCS/src/utils/snapshot.py) (Pemotongan proporsional margin 20%, manual snapshot tombol `s`, auto-snapshot).
- [x] Implementasi [`DetectionLogger`](file:///c:/Users/User/Documents/PROJEK_PYTHON_DETEKSI/DOCS/src/utils/logger.py) (Ekspor ke format CSV & JSONLines secara simultan).
- [x] Unit test tracking, visualizer, snapshot, dan logger di [`tests/test_tracker.py`](file:///c:/Users/User/Documents/PROJEK_PYTHON_DETEKSI/DOCS/tests/test_tracker.py) (5 tests lulus).

#### 📁 File yang Dihasilkan:
* [`src/tracking/tracker.py`](file:///c:/Users/User/Documents/PROJEK_PYTHON_DETEKSI/DOCS/src/tracking/tracker.py)
* [`src/tracking/__init__.py`](file:///c:/Users/User/Documents/PROJEK_PYTHON_DETEKSI/DOCS/src/tracking/__init__.py)
* [`src/utils/visualizer.py`](file:///c:/Users/User/Documents/PROJEK_PYTHON_DETEKSI/DOCS/src/utils/visualizer.py)
* [`src/utils/snapshot.py`](file:///c:/Users/User/Documents/PROJEK_PYTHON_DETEKSI/DOCS/src/utils/snapshot.py)
* [`src/utils/logger.py`](file:///c:/Users/User/Documents/PROJEK_PYTHON_DETEKSI/DOCS/src/utils/logger.py)
* [`src/utils/__init__.py`](file:///c:/Users/User/Documents/PROJEK_PYTHON_DETEKSI/DOCS/src/utils/__init__.py)

---

### 🟢 Fase 4: Antarmuka Aplikasi, Web Dashboard, Tests & Dokumentasi
**Sasaran:** Menyediakan antarmuka interaktif bagi pengguna akhir (Desktop CLI + Web Dashboard Streamlit) serta dokumentasi arsitektur dan panduan pemakaian yang lengkap.

#### ✅ Checklist Pengerjaan:
- [x] Implementasi aplikasi desktop utama [`src/app.py`](file:///c:/Users/User/Documents/PROJEK_PYTHON_DETEKSI/DOCS/src/app.py) dengan keyboard shortcuts real-time (`m`, `t`, `s`, `l`, `b`, `o`, `+`, `-`, `q`).
- [x] Dukungan multi-source: Webcam lokal, file video (.mp4/.avi), gambar statis (.jpg/.png), stream RTSP.
- [x] Implementasi Web Studio modern [`src/web_app.py`](file:///c:/Users/User/Documents/PROJEK_PYTHON_DETEKSI/DOCS/src/web_app.py) menggunakan Streamlit (Image Analysis, Live Camera Input, Log Viewer).
- [x] Penyusunan dokumentasi arsitektur sistem lengkap [`DOCS/ARCHITECTURE.md`](file:///c:/Users/User/Documents/PROJEK_PYTHON_DETEKSI/DOCS/ARCHITECTURE.md).
- [x] Penyusunan panduan penggunaan lengkap [`README.md`](file:///c:/Users/User/Documents/PROJEK_PYTHON_DETEKSI/DOCS/README.md).
- [x] Pembuatan sample image generator untuk pengujian cepat di [`data/samples/sample_face.jpg`](file:///c:/Users/User/Documents/PROJEK_PYTHON_DETEKSI/DOCS/data/samples/sample_face.jpg).
- [x] Verifikasi akhir: Seluruh 13 unit tests Pytest berhasil dijalankan tanpa error.

#### 📁 File yang Dihasilkan:
* [`src/app.py`](file:///c:/Users/User/Documents/PROJEK_PYTHON_DETEKSI/DOCS/src/app.py)
* [`src/web_app.py`](file:///c:/Users/User/Documents/PROJEK_PYTHON_DETEKSI/DOCS/src/web_app.py)
* [`src/__init__.py`](file:///c:/Users/User/Documents/PROJEK_PYTHON_DETEKSI/DOCS/src/__init__.py)
* [`DOCS/ARCHITECTURE.md`](file:///c:/Users/User/Documents/PROJEK_PYTHON_DETEKSI/DOCS/ARCHITECTURE.md)
* [`README.md`](file:///c:/Users/User/Documents/PROJEK_PYTHON_DETEKSI/DOCS/README.md)
* [`DOCS/FASE_PENGEMBANGAN.md`](file:///c:/Users/User/Documents/PROJEK_PYTHON_DETEKSI/DOCS/FASE_PENGEMBANGAN.md) (Dokumen ini)

---

## 🔮 Roadmap Pengembangan Lanjutan (Next Phases / Future Scope)

Jika Anda ingin memperluas sistem ini lebih jauh ke skala industri, berikut adalah rencana fase lanjutan yang dapat diimplementasikan secara modular:

### 🟡 Fase 5: Face Recognition (Pengenalan Identitas Wajah)
- [ ] Integrasi model ekstraksi face embedding (ArcFace / FaceNet / AdaFace).
- [ ] Database vektor wajah lokal (ChromaDB / FAISS / SQLite) untuk registrasi identitas.
- [ ] Pencocokan wajah real-time dengan label nama (misal: "John Doe (98.4%)").

### 🟡 Fase 6: Anti-Spoofing & Liveness Detection
- [ ] Deteksi kedipan mata (*Eye Blink Aspect Ratio - EAR*).
- [ ] Deteksi pola tekstur / refleksi layar HP untuk mencegah pemalsuan foto/video.
- [ ] Status badge liveness pada OSD: `REAL` vs `SPOOF`.

### 🟡 Fase 7: REST API & Microservice Deployment
- [ ] Wrapper FastAPI untuk deteksi via endpoint HTTP / WebSocket.
- [ ] Dockerfile & Docker Compose untuk containerization siap-pakai di server / cloud.
