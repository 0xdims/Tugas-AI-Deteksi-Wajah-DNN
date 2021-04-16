  
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dlib
import cv2
import numpy as np
import os

# Menggunakan library pendeteksi wajah dari dlib
# Ini memungkinkan untuk mendeteksi wajah dalam gambar
pendeteksi_wajah = dlib.get_frontal_face_detector()
# Menggunakan Pose Predictor dari dlib
# untuk mendeteksi titik landmark di wajah dan memahami pose / sudut wajah
prediksi_bentuk = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
# Menggunakan model pengenalan wajah
# Untuk melakukan pengkodean wajah (angka yang mengidentifikasi wajah orang tertentu)
model_pengenalan_wajah = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
# Toleransi untuk perbandingan wajah
# Semakin rendah angkanya - semakin tinggi perbandingannya
# Untuk menghindari kecocokan yang salah, gunakan nilai lebih rendah
# Untuk menghindari false negatif (output wajah yang sama tidak cocok), gunakan nilai yang lebih tinggi
# Nilai 0.5-0.6 dapat bekerja dengan baik
TOLERANSI = 0.47

# Fungsi ini akan mengambil gambar dan mengembalikan pengkodean wajahnya menggunakan neural network
def dapatkan_pengkodean_wajah(path_gambar):
    # Memuat gambar menggunakan cv2
    gambar = cv2.imread(path_gambar)
    # Mendeteksi wajah menggunakan detektor wajah
    wajah_terdeteksi = pendeteksi_wajah(gambar, 1)
    # Mendapatkan pose / landmark wajah
    # Akan digunakan sebagai input yang menghitung pengkodean wajah
    # Ini memungkinkan neural network dapat menghasilkan angka yang sama untuk wajah orang yang sama, terlepas dari sudut kamera dan / atau pemosisian wajah dalam gambar
    bentuk_wajah = [prediksi_bentuk(gambar, wajah) for wajah in wajah_terdeteksi]
    # Menghitung pengkodean wajah setiap wajah yang terdeteksi
    return [np.array(model_pengenalan_wajah.compute_face_descriptor(gambar, pose_wajah, 1)) for pose_wajah in bentuk_wajah]

# Fungsi ini mengambil daftar wajah yang dikenali
def bandingkan_pengkodean_wajah(wajah_diketahui, wajah):
    # Menemukan perbedaan antara setiap wajah yang diketahui dan wajah yang diberikan (yang dibandingkan)
    # Mengitung nilai perbedaan dengan setiap wajah yang diketahui
    # Kembalikan array dengan nilai True/False berdasarkan apakah wajah yang diketahui cocok dengan wajah yang diberikan cocok atau tidak
    # Kecocokan terjadi ketika perbedaan (nilai) antara wajah yang diketahui dan wajah yang diberikan kurang dari atau sama dengan nilai TOLERANSI
    return (np.linalg.norm(wajah_diketahui - wajah, axis=1) <= TOLERANSI)

# Fungsi ini mengembalikan nama orang yang gambarnya cocok dengan wajah yang diberikan (atau 'Wajah tidak ada yang cocok')
def cari_kecocokan(wajah_diketahui, nama, wajah):
    # Panggil bandingkan_pengkodean_wajah untuk mendapatkan daftar nilai True/False yang menunjukkan apakah ada kecocokan atau tidak
    kecocokan = bandingkan_pengkodean_wajah(wajah_diketahui, wajah)
    # Mengembalikan nilai nama
    nilai = 0
    for cocok in kecocokan:
        if cocok:
            return nama[nilai]
        nilai += 1
     # Kembali jika tidak ditemukan ada yang cocok
    return 'Wajah tidak ada yang cocok'

# Mendapatkan directory ke semua gambar yang dikenal
# Memfilter ekstensi .jpg - jadi ini hanya akan berfungsi dengan gambar JPEG yang diakhiri dengan .jpg
namafile_gambar = filter(lambda x: x.endswith('.jpg'), os.listdir('images/'))
# Urutkan dalam urutan abjad
namafile_gambar = sorted(namafile_gambar)
# Dapatkan directory lengkap ke gambar
paths_ke_gambar = ['images/' + x for x in namafile_gambar]
# Daftar pengkodean wajah yang kami miliki
pengkodean_wajah = []
# Perulangan gambar untuk mendapatkan pengkodean satu per satu
for path_gambar in paths_ke_gambar:
    # Dapatkan pengkodean wajah dari gambar
    pengkodean_wajah_dalam_gambar = dapatkan_pengkodean_wajah(path_gambar)
    # Pastikan hanya ada satu wajah dalam gambar
    if len(pengkodean_wajah_dalam_gambar) != 1:
        print("Harap ubah gambar: " + path_gambar + " - ini mempunyai " + str(len(pengkodean_wajah_dalam_gambar)) + " wajah; hanya boleh ada satu")
        exit()
    # Tambahkan pengkodean wajah yang ditemukan pada gambar itu ke daftar pengkodean wajah yang kita miliki
    pengkodean_wajah.append(dapatkan_pengkodean_wajah(path_gambar)[0])

# Dapatkan jalur ke semua gambar uji
# Memfilter ekstensi .jpg - jadi ini hanya akan berfungsi dengan gambar JPEG yang diakhiri dengan .jpg
namafile_test = filter(lambda x: x.endswith('.jpg'), os.listdir('test/'))
# Dapatkan jalur lengkap untuk menguji gambar
path_gambar_test = ['test/' + x for x in namafile_test]
# Dapatkan daftar nama orang dengan menghilangkan ekstensi .JPG dari nama file gambar
nama = [x[:-4] for x in namafile_gambar]
# Perulangan gambar uji untuk menemukan kecocokan satu per satu
for path_gambar in path_gambar_test:
    # Dapatkan pengkodean wajah dari gambar uji
    pengkodean_wajah_dalam_gambar = dapatkan_pengkodean_wajah(path_gambar)
    # Pastikan hanya ada satu wajah dalam gambar
    if len(pengkodean_wajah_dalam_gambar) != 1:
        print("Harap ubah gambar: " + path_gambar + " - ini mempunyai " + str(len(pengkodean_wajah_dalam_gambar)) + " wajah; hanya boleh ada satu")
        exit()
    # Temukan kecocokan untuk pengkodean wajah yang ditemukan di gambar uji ini
    cocok = cari_kecocokan(pengkodean_wajah, nama, pengkodean_wajah_dalam_gambar[0])
    # Cetak directory gambar uji dan hasil yang sesuai
    print(path_gambar, cocok)
