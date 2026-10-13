# Product Requirements Document (PRD)
## Project: Sistem Deteksi Wajah Berbasis Python (Modern Face Detection System)

---

## 1. Ringkasan Eksekutif (Executive Summary)
Dokumen ini mendefinisikan ruang lingkup, kebutuhan fungsional dan non-fungsional, arsitektur teknis, serta evaluasi strategis untuk pembangunan aplikasi **Deteksi Wajah (Face Detection)** menggunakan ekosistem Python.

Tujuan utama dari proyek ini adalah membangun sistem deteksi wajah yang **akurat, berlatensi rendah (real-time), modular, dan siap dikembangkan (production-ready)**, baik untuk kebutuhan analitik video, otomasi, maupun integrasi sistem keamanan.

---

## 2. Masukan & Kritik Senior Programmer (Senior Review & Reality Check)

> *"Banyak proyek computer vision pemula gagal melangkah ke tahap produksi karena salah memilih fondasi sejak awal atau meremehkan kompleksitas data di dunia nyata."*

Berikut adalah catatan kritis, evaluasi risiko, dan saran arsitektur dari sudut pandang engineer berpengalaman:

### ⚠️ Kritik 1: Ambiguitas Definisi Ruang Lingkup
* **Masalah:** Pemula sering kali mencampuradukkan antara:
  1. **Face Detection (Deteksi):** Mengetahui *di mana* lokasi wajah (bounding box `[x, y, w, h]`).
  2. **Face Alignment / Landmarks:** Menentukan titik-titik fitur wajah (mata, hidung, bibir, 68/468 titik).
  3. **Face Recognition (Pengenalan):** Mengetahui *siapa* pemilik wajah tersebut (klasifikasi / face embedding `128-d` / `512-d`).
  4. **Liveness Detection / Anti-Spoofing:** Memastikan wajah tersebut adalah manusia asli, bukan foto di layar HP atau cetakan kertas.
* **Saran Senior:** Mulai dengan fokus solid pada **Face Detection & Tracking** terlebih dahulu. Buat arsitektur modular sehingga modul *Recognition* atau *Liveness* bisa dicolok (*pluggable*) di kemudian hari tanpa merombak ulang *capture pipeline*.

### ⚠️ Kritik 2: Hindari Penggunaan Haar Cascade untuk Kebutuhan Modern
* **Masalah:** Tutorial lama banyak mengajarkan `cv2.CascadeClassifier('haarcascade_frontalface_default.xml')`. Haar Cascade sudah usang (era 2001), sangat rentan terhadap *false positives*, gagal saat wajah miring (*in-plane/out-of-plane rotation*), dan rapuh terhadap variasi pencahayaan.
* **Saran Senior:** Gunakan standar industri modern:
  * **MediaPipe Face Detection (BlazeFace):** Sangat ringan, optimal untuk CPU / laptop tanpa GPU dedicated, latency sangat rendah (~10-20ms).
  * **OpenCV YuNet (DNN):** Sangat cepat (~10ms pada CPU), sudah built-in di OpenCV 4.5.4+, akurat terhadap sudut ekstrem.
  * **YOLOv8-Face / RetinaFace:** Sangat akurat, robust terhadap oklusi dan kerumunan (*crowd detection*), ideal jika didukung GPU atau dioptimasi via ONNX Runtime.

### ⚠️ Kritik 3: Bottleneck Threading pada Video Capture di Python
* **Masalah:** Menjalankan `cap.read()` dari OpenCV di thread yang sama dengan inferensi model akan menyebabkan *frame drop* dan *latency spike*. Kamera terblokir saat AI sedang melakukan kalkulasi.
* **Saran Senior:** Pisahkan pipeline ke dalam pola **Producer-Consumer** (Thread/Process terpisah):
  1. *Thread 1 (Producer):* Menangkap frame kamera secepat mungkin ke buffer / queue non-blocking.
  2. *Thread 2 (Consumer/Worker):* Mengambil frame terbaru, melakukan inferensi model, dan menggambar output.
  3. *Skip Frame Mechanism:* Jika FPS kamera 30 tapi AI hanya mampu 15 FPS, buang frame lama agar tidak terjadi lag waktu nyata.

### ⚠️ Kritik 4: Privasi Data Biometrik & Legalitas (UU PDP / GDPR)
* **Masalah:** Wajah adalah data biometrik sensitif. Menyimpan rekaman video mentah atau gambar wajah tanpa persetujuan dapat melanggar hukum perlindungan data pribadi.
* **Saran Senior:**
  * Implementasikan opsi *Edge Processing* (data diproses di memori tanpa disimpan permanen).
  * Jika memerlukan logging, simpan metadata (timestamp, bounding box coordinates, confidence score) atau enkripsi data wajah.

---

## 3. Kebutuhan Produk (Product Requirements)

### 3.1 Kebutuhan Fungsional (Functional Requirements)
1. **Multi-Source Input Stream:**
   * Mendukung input dari Webcam lokal / USB Camera.
   * Mendukung input dari file video (`.mp4`, `.avi`, `.mkv`).
   * Mendukung input dari file gambar statis (`.jpg`, `.png`).
   * Mendukung input stream RTSP/IP Camera (untuk kebutuhan CCTV/Security).
2. **Core Detection Engine:**
   * Mendeteksi multi-wajah secara simultan dalam satu frame.
   * Menyediakan *Confidence Threshold Slider / Config* (misal: hanya deteksi dengan confidence > 0.6 yang ditampilkan).
   * Menghasilkan Bounding Box akurat dan titik landmark utama (mata, hidung, telinga).
3. **Face Tracking & Unique ID Assignment:**
   * Mampu melacak pergerakan wajah antar-frame (menggunakan algoritma seperti ByteTrack atau IoU Tracker) dan memberikan `Face ID` unik (mencegah double-counting).
4. **Visual Overlay & OSD (On-Screen Display):**
   * Menampilkan Bounding Box dengan warna dinamis berdasarkan confidence/status.
   * Menampilkan metrics real-time: FPS, Deteksi Total Wajah, Resolusi Frame, Waktu Inferensi (ms).
5. **Export & Logging:**
   * Opsi untuk menyimpan screenshot snapshot wajah yang terdeteksi.
   * Opsi ekspor log kejadian ke file CSV / JSON (timestamp, face ID, confidence, bounding box).

### 3.2 Kebutuhan Non-Fungsional (Non-Functional Requirements)
1. **Performa & Kecepatan:**
   * Target FPS minimum: **25 - 30+ FPS** pada CPU standar (menggunakan MediaPipe / YuNet / quantized model).
   * Inference latency: **< 30ms** per frame.
2. **Modularitas & Clean Code:**
   * Mengikuti prinsip OOP & SOLID. Kode detektor harus berupa abstraksi kelas (`BaseDetector -> MediaPipeDetector / YuNetDetector`) sehingga model mudah ditukar ganti.
3. **Kemudahan Instalasi:**
   * Pengelolaan dependensi yang bersih via `requirements.txt` atau `pyproject.toml` dengan virtual environment (`venv`).
4. **Cross-Platform:**
   * Kompatibel dijalankan di Windows, Linux, dan macOS.

---

## 4. Perbandingan Rekomendasi Tech Stack

| Model / Library | Kecepatan (CPU) | Akurasi | Deteksi Wajah Miring / Jauh | Rekomendasi Penggunaan |
| :--- | :--- | :--- | :--- | :--- |
| **Haar Cascade** | Sangat Cepat | Sangat Rendah | Buruk | ❌ *Tidak disarankan (Outdated)* |
| **OpenCV YuNet (DNN)** | Sangat Cepat (~10ms) | Tinggi | Bagus | ✅ **Sangat Direkomendasikan untuk CPU Standar** |
| **MediaPipe Face** | Sangat Cepat (~15ms) | Sangat Tinggi | Sangat Bagus | ✅ **Sangat Direkomendasikan (MVP & Fleksibilitas)** |
| **YOLOv8-Face** | Sedang-Cepat (Butuh GPU/ONNX) | Luar Biasa | Sangat Bagus (Kerumunan) | ✅ **Terbaik untuk Skala Industri / GPU** |
| **RetinaFace** | Sedang (Berat di CPU) | State-of-the-Art | Terbaik | ⚠️ *Gunakan jika akurasi absolut > kecepatan* |

### Rekomendasi Stack Utama:
* **Bahasa:** Python 3.10+ (atau 3.11/3.12)
* **Core Vision Library:** `opencv-python`, `numpy`
* **Inference Engine:** `mediapipe` dan `opencv-contrib-python` (YuNet) / `onnxruntime`
* **GUI / Antarmuka (Opsional):**
  * *Tahap 1 (PoC):* OpenCV Window (`cv2.imshow`)
  * *Tahap 2 (Dashboard Interaktif):* `Streamlit` / `Gradio` (Web UI cepat) atau `PyQt6 / PySide6` (Desktop App mandiri).

---

## 5. Struktur Folder Proyek yang Direkomendasikan (Standard Production Layout)

```text
PROJEK_PYTHON_DETEKSI/
│
├── DOCS/
│   ├── PRD.MD                   # Dokumen kebutuhan produk & catatan senior
│   └── ARCHITECTURE.md          # Dokumen diagram & alur data sistem
│
├── src/
│   ├── __init__.py
│   ├── core/                    # Engine inti dan abstraksi
│   │   ├── __init__.py
│   │   ├── base_detector.py     # Abstract Base Class untuk detektor
│   │   ├── mediapipe_detector.py# Implementasi MediaPipe
│   │   └── yunet_detector.py    # Implementasi OpenCV YuNet
│   │
│   ├── video/                   # Pipeline streaming video
│   │   ├── __init__.py
│   │   ├── video_stream.py      # Threaded video capture
│   │   └── fps_counter.py       # Helper penghitung FPS akurat
│   │
│   ├── tracking/                # Modul pelacakan objek/wajah
│   │   ├── __init__.py
│   │   └── tracker.py
│   │
│   ├── utils/                   # Helper penggambaran visual & logging
│   │   ├── __init__.py
│   │   ├── visualizer.py        # Gambar bounding box, landmark, OSD
│   │   └── logger.py            # Log detector events
│   │
│   └── app.py                   # Entry point aplikasi utama
│
├── tests/                       # Unit test & benchmark
│   ├── test_detector.py
│   └── test_stream.py
│
├── data/                        # Asset sample untuk testing (tidak di-commit jika besar)
│   ├── samples/                 # Contoh video / foto
│   └── models/                  # File bobot model (e.g. face_detection_yunet.onnx)
│
├── .gitignore                   # Mengabaikan venv, cache, model besar
├── requirements.txt             # Daftar dependensi library
└── README.md                    # Panduan instalasi dan cara menjalankan
```

---

## 6. Rencana Tahapan Pengembangan (Development Roadmap)

### Fase 1: Fondasi & MVP (Minimum Viable Product)
- [ ] Setup Virtual Environment (`.venv`) dan `requirements.txt`.
- [ ] Implementasi `BaseDetector` dan `MediaPipeDetector` / `YuNetDetector`.
- [ ] Buat single-thread visualizer sederhana dengan webcam feed.
- [ ] Evaluasi akurasi dan kemudahan setup.

### Fase 2: Optimasi Kinerja & Threaded Stream
- [ ] Implementasi `ThreadedVideoStream` untuk menghindari *capture lag*.
- [ ] Implementasi kalkulasi FPS dan waktu inferensi yang stabil.
- [ ] Tambahkan konfigurasi parameter (resolusi kamera, threshold deteksi, flip camera).

### Fase 3: Fitur Lanjutan (Tracking & Logging)
- [ ] Integrasi Face Tracking (pemberian ID unik pada setiap wajah).
- [ ] Fitur Face Snapshot (auto-save crop wajah dengan kualitas terbaik).
- [ ] Ekspor log deteksi ke JSON/CSV.

### Fase 4: Packaging & Antarmuka
- [ ] Buat UI interaktif (PyQt/PySide6 untuk desktop atau Streamlit untuk browser).
- [ ] Tambahkan automated tests (unit tests).
- [ ] Dokumentasi lengkap di `README.md`.

---

## 7. Metrik Keberhasilan (Key Performance Indicators)
1. **Framerate:** Minimal 28 FPS stabil pada resolusi 720p (HD) di CPU standar.
2. **False Positive Rate:** < 3% pada kondisi ruangan normal.
3. **Response Time / Latency:** < 35ms dari penangkapan frame hingga visualisasi di layar.
4. **Code Quality:** Arsitektur modular, *type hinting* lengkap, mudah diperluas ke Face Recognition di masa depan.
