import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import pyfirmata
import time

# Inisialisasi koneksi ke Wemos D1 Mini melalui protokol Firmata
board = pyfirmata.Arduino("COM5")  # Ganti "COM5" dengan port USB Wemos D1 Mini Anda
led1 = board.get_pin('d:3:o')  # LED 1 terhubung ke pin D3
led2 = board.get_pin('d:5:o')  # LED 2 terhubung ke pin D5

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300

labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
          , "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape
        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            # print(prediction, index)
            print(labels[index])

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        # Menyalakan LED berdasarkan hasil klasifikasi
        if labels[index] == "1":
            led1.write(1)  # Nyalakan LED 1
            led2.write(0)  # Matikan LED 2
            print(labels[1])
        elif labels[index] == "2":
            led1.write(0)  # Matikan LED 1
            led2.write(1)  # Nyalakan LED 2
            print(labels[2])
        else:
            led1.write(0)  # Matikan LED 1
            led2.write(0)  # Matikan LED 2

        # Tampilkan hasil klasifikasi pada layar
        cv2.rectangle(imgOutput, (x - offset, y - offset-50),
                      (x - offset+90, y - offset-50+50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y -26), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x-offset, y-offset),
                      (x + w+offset, y + h+offset), (255, 0, 255), 4)

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1)
    # Keluar dari loop jika tombol 'c' ditekan
    if key == ord('c'):
        cap.release()
        cv2.destroyAllWindows()
        led1.write(0)  # Matikan LED 1 sebelum keluar
        led2.write(0)  # Matikan LED 2 sebelum keluar
        board.exit()  # Keluar dari koneksi Firmata
        break