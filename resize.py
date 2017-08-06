import cv2
import numpy as np
import scipy as sp
import os
from PIL import Image	
import matplotlib.pyplot as plt
import matplotlib

#Reading all the images from the folder
path = "C:/GMU/FourthSem/CS682/bounding/"
image_names = os.listdir(path)

#Creating the image path for reading
image_paths = []
for p in image_names:
	image_paths1 = os.path.join("C:/GMU/FourthSem/CS682/bounding/",p)
	image_paths.append(image_paths1)
#print image_paths

#Height and Width of the final image
width = 50
height = 75

idx=0	
os.mkdir('C:/GMU/FourthSem/CS682/resize',0755)
#####for the folder data
for img1_path in image_paths:
	img_gray = cv2.imread(img1_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)	
	idx+=1
	h,w = img_gray.shape
	
	#Calculating the position for image patching
	position_w = int((width - w) / 2) 
	position_h = int((height - h) / 2) 

	#Image of size 50x75 and with uniform background
	new_image = []                                             
	for i in range (0, height):                               
		new = []                 
		for j in range (0, width):    
			new.append(207)              
		new_image.append(new) 

	#print new_image

	if w > width or h > height:
		dim = (50, 75)
		resize = cv2.resize(img_gray, dim, interpolation=cv2.INTER_LINEAR)
		cv2.imwrite('C:/GMU/FourthSem/CS682/resize/' +str(idx) + '.jpg',resize)
	else:
		new_w = position_w + 1
		new_h = position_h + 1
		for i in range(0,h):
			new_w = position_w + 1
			for j in range(0,w):
				new_image[new_h][new_w] = img_gray[i,j]
				if new_w<width:
					new_w = new_w + 1
				else:
					continue
			if new_h < height:
				new_h = new_h + 1
			else:
				continue
		final_image = np.array(new_image, dtype = np.uint8)
		cv2.imwrite('C:/GMU/FourthSem/CS682/resize' +str(idx) + '.jpg',final_image)	
	
	
	

cv2.waitKey(0)
cv2.destroyAllWindows()