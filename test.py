import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300

# folder = "Data/C"
# counter = 0

labels = ["right_0", "right_1", "right_2", "right_3", "right_4", "right_5", "right_6", "right_7", "right_8", "right_9", "right_A", "right_B", "right_C", "right_D", "right_E", "right_F", "right_G", "right_H", "right_I", "right_J", "right_K", "right_L", "right_M", "right_N", "right_O", "right_P", "right_Q", 
          "right_R", "right_S", "right_T", "right_U", "right_V", "right_W", "right_X", "right_Y", "right_Z", "left_0", "left_1", "left_2", "left_3", "left_4", "left_5", "left_6", "left_7", "left_8", "left_9", "left_A", "left_B", "left_C", "left_D", "left_E", "left_F", "left_G", "left_H", 
          "left_I", "left_J", "left_K", "left_L", "left_M", "left_N", "left_O", "left_P", "left_Q", "left_R", "left_S", "left_T", "left_U", "left_V", "left_W", "left_X", "left_Y", "left_Z"]

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
            print(prediction, index)

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        cv2.rectangle(imgOutput, (x - offset, y - offset-50),
                      (x - offset+90, y - offset-50+50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y -26), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x-offset, y-offset),
                      (x + w+offset, y + h+offset), (255, 0, 255), 4)

        # cv2.imshow("ImageCrop", imgCrop)
        # cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1)
    # Keluar dari loop jika tombol 'c' ditekan
    if key == ord('c'):
        cap.release()
        cv2.destroyAllWindows()
        break