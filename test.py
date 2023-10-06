import cv2 #Ini adalah OpenCV, sebuah library untuk pengolahan gambar dan video.
from cvzone.HandTrackingModule import HandDetector # Ini adalah modul yang digunakan untuk mendeteksi tangan dan mengklasifikasikan isyarat tangan.
from cvzone.ClassificationModule import Classifier # Ini adalah modul yang digunakan untuk mendeteksi tangan dan mengklasifikasikan isyarat tangan.
import numpy as np # Library untuk operasi numerik di Python.
import math # Library untuk operasi matematika di Python.
import pyfirmata # Library untuk berkomunikasi dengan papan mikrokontroler Wemos D1 Mini melalui protokol Firmata.
# import time

# Inisialisasi koneksi ke Wemos D1 Mini melalui protokol Firmata
board = pyfirmata.Arduino("COM5")  # Ganti "COM5" dengan port USB Wemos D1 Mini Anda
led1 = board.get_pin('d:3:o')  # LED 1 terhubung ke pin D3
# led2 = board.get_pin('d:5:o')  # LED 2 terhubung ke pin D5

# # Inisialisasi pengambilan gambar dari kamera
cap = cv2.VideoCapture(0) # Mengambil video dari kamera yang terhubung ke komputer
detector = HandDetector(maxHands=1) # Menginisialisasi detektor tangan
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt") # Menginisialisasi model klasifikasi gestur tangan

offset = 20 # Margin yang akan ditambahkan ke gambar tangan yang diambil
imgSize = 300 # Ukuran gambar tangan yang akan digunakan untuk klasifikasi

# Daftar label yang akan digunakan untuk hasil klasifikasi
labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
          , "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

while True:
    # Membaca frame dari kamera
    success, img = cap.read()
    # Membuat salinan independen dari gambar asli.
    imgOutput = img.copy()
    
    # Mendeteksi tangan dalam gambar
    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox'] # Mendapatkan kotak pembatas (bounding box) tangan
        # Variabel x adalah koordinat x bagian atas kiri kotak pembatas
        # y adalah koordinat y bagian atas kiri
        # w adalah lebar kotak
        # h adalah tinggi kotak.

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        # Membuat gambar imgWhite yang berukuran imgSize x imgSize pixel dan berisi warna putih (semua piksel bernilai 255). 
        # Ini adalah latar belakang gambar yang akan digunakan untuk menggabungkan gambar tangan yang diproses.
        
        # Memotong gambar tangan yang terdeteksi
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        # Memotong gambar tangan dari gambar asli dengan menambahkan margin sebesar offset. 
        # Ini dilakukan untuk memastikan bahwa tangan tidak terlalu dekat dengan batas gambar.

        imgCropShape = imgCrop.shape
        # Mendapatkan dimensi gambar tangan yang dipotong, terutama lebar dan tingginya.
        
        aspectRatio = h / w # Menghitung rasio aspek (aspect ratio) tangan.
        # Ini digunakan untuk menentukan apakah tangan lebih tinggi atau lebih lebar. 

        if aspectRatio > 1: # Ini adalah cabang jika tangan lebih tinggi dari lebar.
            k = imgSize / h # Menghitung faktor perbesaran (scaling factor) berdasarkan tinggi gambar imgCrop dibagi dengan imgSize.
            wCal = math.ceil(k * w) # Menghitung lebar yang diubah ukurannya berdasarkan faktor perbesaran. 
            # Nilai ini dibulatkan ke atas menjadi bilangan bulat.
            
            imgResize = cv2.resize(imgCrop, (wCal, imgSize)) # Mengubah ukuran gambar tangan menjadi imgSize x wCal.
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            # Menghitung seberapa banyak margin yang harus ditambahkan di sekitar gambar yang diubah ukurannya.
            
            imgWhite[:, wGap:wCal + wGap] = imgResize
            # Menggabungkan gambar yang telah diubah ukuran dengan latar belakang putih imgWhite. 
            # Ini dilakukan dengan mengganti piksel di antara wGap dan wCal + wGap di baris-baris semua (:) 
            # kolom pada imgWhite dengan piksel dari imgResize.
            
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            # Mengeklasifikasikan gambar tangan yang telah diubah ukuran dengan menggunakan model klasifikasi 
            # yang telah diinisialisasi sebelumnya.
            
            # print(prediction, index)
            print("yang terdeteksi : " + labels[index])
            
        # cabang jika tangan lebih lebar dari tinggi
        # berfokus pada lebar gambar tangan dan menghitung tinggi gambar yang diubah ukurannya.
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        # Menyalakan LED berdasarkan hasil klasifikasi
        if labels[index] == "5":
            led1.write(1)  # Nyalakan LED 1
            # led2.write(0)  # Matikan LED 2
            # print(labels[1])
        # elif labels[index] == "2":
        #     led1.write(0)  # Matikan LED 1
        #     led2.write(1)  # Nyalakan LED 2
        #     # print(labels[2])
        else:
            led1.write(0)  # Matikan LED 1
            # led2.write(0)  # Matikan LED 2

        # Tampilkan hasil klasifikasi pada layar
        cv2.rectangle(imgOutput, (x - offset, y - offset-50),
                      (x - offset+90, y - offset-50+50), (255, 0, 255), cv2.FILLED)
        # kode untuk menggambar kotak latar belakang berisi label hasil klasifikasi di atas tangan yang terdeteksi.
        # imgOutput: Gambar yang akan digunakan sebagai dasar untuk menggambar kotak dan teks.
        # (x - offset, y - offset-50): Titik awal (kiri atas) dari kotak, dihitung berdasarkan x dan y tangan dengan penambahan dan pengurangan offset. Offset digunakan untuk menyesuaikan letak kotak.
        # (x - offset+90, y - offset-50+50): Titik akhir (kanan bawah) dari kotak, dengan lebar dan tinggi sebesar 90 dan 50 piksel, masing-masing.
        # (255, 0, 255): Warna kotak dalam format BGR (Biru, Hijau, Merah). Di sini, warna yang digunakan adalah magenta (biru + merah).
        # cv2.FILLED: Parameter ini menandakan bahwa kotak harus diisi dengan warna, bukan hanya garis tepinya.

        cv2.putText(imgOutput, labels[index], (x, y -26), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        # kode yang menambahkan teks label hasil klasifikasi ke gambar.
        # imgOutput: Gambar yang akan digunakan sebagai dasar untuk menambahkan teks.
        # labels[index]: Label hasil klasifikasi yang akan ditampilkan. index adalah hasil klasifikasi yang telah diambil sebelumnya dari model.
        # (x, y - 26): Koordinat teks, dihitung berdasarkan x dan y tangan dengan penambahan offset. Teks akan ditampilkan sedikit di atas kotak.
        # cv2.FONT_HERSHEY_COMPLEX: Jenis font yang digunakan untuk teks.
        # 1: Skala / besar font.
        # (255, 255, 255): Warna teks dalam format BGR (putih).
        # 2: Ketebalan garis teks.
        
        cv2.rectangle(imgOutput, (x-offset, y-offset),
                      (x + w+offset, y + h+offset), (255, 0, 255), 4)
        # kode yang menggambar kotak pembatas tangan yang terdeteksi pada gambar.
        # imgOutput: Gambar yang akan digunakan sebagai dasar untuk menggambar kotak.
        # (x-offset, y-offset): Titik awal (kiri atas) dari kotak pembatas tangan, dihitung berdasarkan x dan y tangan dengan penambahan dan pengurangan offset. Offset digunakan untuk memperluas kotak pembatas.
        # (x + w + offset, y + h + offset): Titik akhir (kanan bawah) dari kotak pembatas tangan, dengan penambahan offset di sekitar x, y, w, dan h.
        # (255, 0, 255): Warna kotak pembatas dalam format BGR (magenta).
        # 4: Ketebalan garis kotak.

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1)
    # Keluar dari loop jika tombol 'c' ditekan
    if key == ord('c'):
        cap.release()
        cv2.destroyAllWindows()
        led1.write(0)  # Matikan LED 1 sebelum keluar
        # led2.write(0)  # Matikan LED 2 sebelum keluar
        board.exit()  # Keluar dari koneksi Firmata
        break