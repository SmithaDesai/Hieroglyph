import cv2
import numpy as np
import scipy as sp
from PIL import Image	
import matplotlib.pyplot as plt
import matplotlib
import os


img1_path = 'C:/GMU/FourthSem/CS682/input/egyptianTexts20.jpg'

#Reading the image
img_gray = cv2.imread(img1_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
#img_gray = img_gray[600:1583,0:200]

#Thresholding the image
thresh, img = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
#cv2.imshow('threshold', img)

#Eroding the image
kernel = np.ones((3,3), np.uint8)
img_erosion = cv2.erode(img, kernel, iterations=1)
#cv2.imshow('erosion', img_erosion)

#To invert the colors in binary image
img_not = cv2.bitwise_not(img_erosion)
#cv2.imshow('not', img_not)

outputImage = img_gray.copy()

#Calculating the contours
contours, hierarchy = cv2.findContours(img_not, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#Drawing bounding boxes
idx=0
boxes =[]
width = 64
height = 128
os.mkdir('C:/GMU/FourthSem/CS682/bounding',0755)
for c in contours:
	idx += 1
	#print idx
	rect = cv2.boundingRect(c)
	if rect[2] < 15 or rect[3] < 15: continue
	
	x,y,w,h = rect
	roi=img_gray[y:y+h,x:x+w]
	cv2.rectangle(img_gray,(x,y),(x+w,y+h),(0,255,0),2)
	#cv2.imwrite('C:/GMU/FourthSem/CS682/bounding/' +str(idx) + '.jpg',roi)
#cv2.imwrite('C:/GMU/FourthSem/CS682/bounding.jpg',img_gray)

cv2.waitKey(0)
cv2.destroyAllWindows()