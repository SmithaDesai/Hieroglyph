import cv2 
import numpy as np
import os

from skimage.feature import hog

# Function To find the keypoints using HOG descriptor
def keypoints(image, isTrain):

    # Read the image in grayscale
    img = cv2.imread(image,0)

    # Get the HOG descriptor
    fd, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualise=True)


    # Reshape the matrix to be used in BOW
    des = np.reshape(fd, (-1, len(fd)))
    return des

# Function to return the list of files with the complete path
def createList(path):
    return [os.path.join(path, f) for f in os.listdir(path)]
