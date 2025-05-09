# Proyek Akhir Pengolahan Sinyal Digital

## Tentang Proyek
Proyek ini merupakan tugas besar mata kuliah Pengolahan Sinyal Digital yang mengimplementasikan sistem pengukuran sinyal respirasi dan sistem pengukuran remote-photopletysmography (rPPG). Program ini mengambil input video dari webcam secara real-time, memproses sinyal-sinyal vital, dan menampilkan hasil pengukuran menggunakan teknik pengolahan sinyal digital.

## Fitur
- (Soon)

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
- [Nama Anggota 2] - [NIM] 
- [Nama Anggota 3] - [NIM] 

## Dosen Pengampu
- [Nama Dosen]

## Lisensi
[Jenis Lisensi] - Lihat file LICENSE untuk detail lebih lanjut.