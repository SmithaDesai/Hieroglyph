import cv2 
import numpy as np
import argparse as arg
import os
import f_utils
from scipy.cluster.vq import *
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler

# print the array completely
np.set_printoptions(threshold='nan')

# Get the values from the command line
par = arg.ArgumentParser()
par.add_argument("-t", "--trainingSet", help="Trainig Set Path", required="True")
arguments = vars(par.parse_args())

# Gives the file names inside the given directory
trainDir = arguments["trainingSet"]
namesTrain = os.listdir(trainDir)

# Initialize the variables
imgPath = []    # Variable to store the image path
imgCls = []     # Variable to store the image class/labels
clsid = 0       # Variable to store the id for the class

# Iterate through the list train images and update the
# image path, image class and class id
for names in namesTrain:
    dir = os.path.join(trainDir, names)
    print dir
    clsPath = f_utils.createList(dir)
    imgPath += clsPath
    imgCls += [clsid] * len(clsPath)
    clsid += 1

# Variable to store the list of all the descriptor
list_des = []   

# Iterate through the image path, get the
# descriptors and append the the list
for images in imgPath:
    des = f_utils.keypoints(images, True)
    list_des.append((images, des))

# Variable to count the number of files in the
# list
fileCount = 1

# Make the descriptor list vertical
descrptr = list_des[0][1]
for image, d in list_des[1:]:
    descrptr = np.vstack((descrptr, d))
    fileCount += 1

# Variable to store the k value to cluster
k_val = 180

# Use k-means clustering by passig
# the descriptor and the k value
v, var = kmeans(descrptr, k_val, 1)

# Initialize with 0.0
fet = np.zeros((len(imgPath), k_val), "float32")

# Iterate through the image path and
# compute the histogram 
for i_iter in xrange(len(imgPath)):
    word, dist = vq(list_des[i_iter][1], v)
    for j_iter in word:
        fet[i_iter][j_iter] += 1

# Perform "term frequence-inverse document frequency"
# to determine the importance of a feature
freq = np.sum((fet > 0)* 1, axis = 0)
invFreq = np.array(np.log((1.0 * len(imgPath) + 1) / (1.0 * freq + 1)), "float32")

# Compute the mean and standard deviation
wordSl = StandardScaler().fit(fet)
# Perform Standarization by centering and scaling
fet = wordSl.transform(fet)

# Linear Support Vector Classification (uses liblinear)
# Create the classifier using the feature and image class
linSvm = LinearSVC()
linSvm.fit(fet, np.array(imgCls))

# Store the classifier in a pickle file to be used later
# for testing
joblib.dump((linSvm, namesTrain, wordSl, k_val, v), "bag.pkl", compress=3)



