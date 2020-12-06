import cv2
import numpy as np

img = cv2.imread('/home/adityan/EE599_Deep_Learning_Project/src/data/images/3_frame134.png')
imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower = np.array([20,0,0])
upper = np.array([100, 255, 255])
imgRange =cv2.inRange(imgHSV, lower, upper)

kernel_noise = np.ones((3,3), np.uint8)
kernel_dilate = np.ones((30,30), np.uint8)
kernel_erode = np.ones((30,30), np.uint8)

imgErode = cv2.erode(imgRange, kernel_noise, 1)
imgDilate = cv2.dilate(imgErode, kernel_dilate, 1)
imgErode = cv2.erode(imgDilate, kernel_erode, 1)

res = cv2.bitwise_and(imgRGB, imgRGB, mask=imgErode)
cv2.imwrite('tmp_2.png', imgRange)
cv2.imwrite('tmp.png', res)
