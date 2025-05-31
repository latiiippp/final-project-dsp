# Proyek Akhir Pengolahan Sinyal Digital

## Tentang Proyek
Proyek ini merupakan tugas besar mata kuliah Pengolahan Sinyal Digital yang mengimplementasikan sistem pengukuran sinyal respirasi dan sistem pengukuran remote-photopletysmography (rPPG). Program ini mengambil input video dari webcam secara real-time, memproses sinyal-sinyal vital, dan menampilkan hasil pengukuran menggunakan teknik pengolahan sinyal digital.

## Alur Pemrosesan Sinyal

### 1. Sinyal Respirasi (Shoulder Movement)
Sinyal respirasi diekstrak menggunakan analisis pergerakan bahu menggunakan Lucas-Kanade Optical Flow. Berikut adalah alur pemrosesan sinyalnya:

1. **Pre-processing**:
   - Deteksi landmark bahu menggunakan MediaPipe Pose
   - Ekstraksi ROI (Region of Interest) untuk bahu kiri dan kanan
   - Tracking fitur menggunakan algoritma Lucas-Kanade Optical Flow

2. **Ekstraksi Sinyal Raw**:
   - Perhitungan rata-rata posisi Y dari fitur yang ditrack
   - Sampling rate: 30 Hz (sesuai FPS kamera)

3. **Filtering Pipeline**:
   - **Detrending**: Menghilangkan drift jangka panjang pada sinyal
   - **Median Filter**:
     - Window size: 0.5 detik
     - Tujuan: Menghilangkan spike noise dan outlier
   - **Bandpass Filter**:
     - Butterworth orde 2
     - Frekuensi cutoff: 0.1-0.5 Hz (6-30 respirasi/menit)
     - Implementasi: Forward-backward filtering (scipy.signal.filtfilt)

4. **Peak Detection**:
   - Normalisasi amplitudo
   - Adaptive prominence threshold (60% dari mean amplitude)
   - Constraint lebar puncak: 0.1-2.0 detik
   - Minimum jarak antar puncak: sesuai frekuensi maksimum respirasi

5. **Kalkulasi RPM**:
   - Moving average pada interval antar puncak (window: 3 interval)
   - Range clipping: 6-25 RPM
   - Update rate: Real-time

### 2. Sinyal rPPG (Remote PhotoPlethysmoGraphy)
Sinyal rPPG diekstrak dari perubahan warna kulit menggunakan metode Plane Orthogonal to Skin (POS). Berikut alur pemrosesannya:

1. **Pre-processing**:
   - Deteksi wajah menggunakan MediaPipe Face Mesh
   - Ekstraksi ROI dahi (20-80% width, 10-25% height dari area wajah)
   - Ekstraksi komponen warna RGB

2. **Ekstraksi Sinyal Raw**:
   - Rata-rata spasial RGB dari ROI
   - Buffer length: 15 detik
   - Sampling rate: 30 Hz

3. **POS Algorithm**:
   - Proyeksi sinyal RGB ke arah ortogonal terhadap variasi warna kulit
   - Window temporal: 1.6 detik
   - Normalisasi untuk mengatasi variasi pencahayaan

4. **Filtering Pipeline**:
   - **Detrending**: Menghilangkan komponen DC dan trend
   - **Bandpass Filter**:
     - Butterworth orde 3
     - Frekuensi cutoff: 0.75-3.0 Hz (45-180 BPM)
     - Zero-phase forward and reverse filtering

5. **Kalkulasi Heart Rate**:
   - Peak detection dengan prominence threshold adaptif
   - Validasi interval temporal
   - Moving average untuk stabilitas pembacaan
   - Range: 45-180 BPM

## Fitur
- Tracking pergerakan bahu real-time menggunakan Lucas-Kanade Optical Flow
- Ekstraksi sinyal rPPG dari video wajah menggunakan algoritma POS
- Visualisasi real-time untuk sinyal raw dan filtered
- Estimasi RPM (Respirasi Per Menit) dan BPM (Beat Per Menit)
- Recording dan plotting hasil analisis

## Persyaratan Sistem
- Python 3.10
- Library terkait (lihat bagian Instalasi)

## Instalasi

1. Clone repositori ini:
```bash
git clone https://github.com/username/final-project-dsp.git
cd final-project-dsp
```

2. Pasang dependensi yang diperlukan:
```bash
pip install -r requirements.txt
```

## Penggunaan
Jalankan file `main.py` untuk memulai aplikasi:
```bash
python main.py
```

Setelah program berjalan:
1. Webcam akan otomatis aktif dan mulai merekam
2. Program akan mendeteksi wajah dan region of interest untuk pengukuran rPPG
3. Secara bersamaan, program akan mendeteksi gerakan untuk analisis sinyal respirasi
4. Hasil pengukuran sinyal respirasi dan rPPG akan ditampilkan secara real-time
5. Tekan tombol 'q' untuk keluar dari program

## Dependensi Utama
- NumPy (1.26.4): Numeric operations dan array processing
- OpenCV (4.11.0): Image processing dan analysis
- MediaPipe (0.10.21): Pose dan landmark detection
- Matplotlib (3.10.3): Data dan results visualization

## Struktur Proyek
```
final-project-dsp/

```

## Tim Pengembang
- Alfajar - 122140122 
- Ikhsannudin Lathief - 122140137 
- Muhammad Ghiffari Iskandar - 122140122

## Dosen Pengampu
- Martin Clinton Tosima Manullang, Ph.D

## Lisensi
