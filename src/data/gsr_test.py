import cv2
import numpy as np
import os 

def show_result(winname, img, wait_time):
	scale = 0.2
	disp_img = cv2.resize(img, None, fx=scale, fy=scale)
	cv2.imshow(winname, disp_img)
	cv2.waitKey(wait_time)

files = os.listdir('/home/adityan/EE599_Deep_Learning_Project/src/data/pewds/')
files = [os.path.join('/home/adityan/EE599_Deep_Learning_Project/src/data/pewds/',file) for file in files]

for file in files:

	img = cv2.resize(cv2.imread(file),(1280,960))
	bg = cv2.resize(cv2.imread('/home/adityan/Downloads/bg.jpg'),(1280,960))

	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	lower_green = np.array([50, 80, 80])
	upper_green = np.array([90, 255, 255])
	mask = cv2.inRange(hsv, lower_green, upper_green)
	mask = cv2.bitwise_not(mask)
	mask = cv2.medianBlur(mask, 7)
	img[mask==0]=[0,0,0]
	bg[mask!=0]=[0,0,0]
	result = bg + img
	#res = cv2.bitwise_and(mask, img)
	#res = cv2.bitwise_or(img, bg)
	#show_result('img', img, 0)
	#show_result('mask',img,2)
	#show_result('bg',bg,2)
	show_result('result', result,0)
	cv2.destroyAllWindows()
