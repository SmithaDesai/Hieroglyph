import cv2 
import numpy as np
import argparse as arg
import os
import f_utils
import itertools
from scipy.cluster.vq import *
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Function to plot the confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Read the values from the pickle file and store in variables
# Classifier, Class Name, Scaler config, k value, vocabulary
clfr, names, slr, k_val, vocblry = joblib.load("bag.pkl")

# Get the values from the command line
par = arg.ArgumentParser()
grp = par.add_mutually_exclusive_group(required=True)
grp.add_argument("-t", "--testSet", help="Trainig Set Path")
grp.add_argument("-i", "--image", help="Path to image")
par.add_argument('-v', "--v", action="store_true")
argument = vars(par.parse_args())

imgPath = []
if argument["testSet"]:
    temp = argument["testSet"]

    try:
        temp_names = os.listdir(temp)
    except OSError:
        print "Directory does not exist"
        exit()

    # Get the test imagees and store in the list
    for name_temp in temp_names:
        dir = os.path.join(temp, name_temp)
        clsPath = f_utils.createList(dir)
        imgPath += clsPath
else:
    imgPath = [argument["image"]]

# Variable to store the list of all the descriptor
list_des = []

# Iterate through the image path, get the
# descriptors and append the the list
for images in imgPath:
    des = f_utils.keypoints(images, True)
    list_des.append((images, des))

# Make the descriptor list vertical
descrptr = list_des[0][1]
for image, d in list_des[0:]:
    descrptr = np.vstack((descrptr, d))

# Initialize with 0.0
fet = np.zeros((len(imgPath), k_val), "float32")

# Iterate through the image path and
# compute the histogram
for i_iter in xrange(len(imgPath)):
    word, dist = vq(list_des[i_iter][1], vocblry)
    for j_iter in word:
        fet[i_iter][j_iter] += 1

# Perform "term frequence-inverse document frequency"
# to determine the importance of a feature
freq = np.sum((fet > 0)* 1, axis = 0)
invFreq = np.array(np.log((1.0 * len(imgPath) + 1) / (1.0 * freq + 1)), "float32")

# Perform Standarization by centering and scaling
fet = slr.transform(fet)

# Get the prediction data and store 
pred = [names[i] for i in clfr.predict(fet)]

imglist = []    # Variable to store the image list
predlist = []   # Variable to store the prediction list

if argument["v"]:
    # Iterate through the list and store the image path and
    # prediction in two different list
    for image_path, prediction in zip(imgPath, pred):
        imglist.append(image_path)
        predlist.append(prediction)
        print os.path.basename(image_path) + "\t" + prediction


# To calculate the accuracy rate
counter = 0
actClas = []
for i in range(len(imglist)):
    nme = imglist[i]
    pos1 = nme.find("_")
    pos2 = nme.find(".")
    actClas.append(nme[(pos1+1):pos2])
    
    if (nme[(pos1+1):pos2] == predlist[i]):
        counter += 1
##print "Number of matches: " + str(counter)
##print "Number of test images: " + str(len(imglist))
##print "Accuracy Rate: " + str(float(counter)/len(imglist) * 100)


# plot confusion matrix

# Get the Train classes
train_cls = os.listdir("train")

# Compute confusion matrix
cnf_matrix = confusion_matrix(actClas, predlist)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
##plt.figure()
##plot_confusion_matrix(cnf_matrix, classes=train_cls,
##                      title='Confusion matrix, without normalization')
##
##plt.show()

FP = np.sum(cnf_matrix, axis=0) - np.diag(cnf_matrix)  
FN = np.sum(cnf_matrix, axis=1) - np.diag(cnf_matrix)
TP = np.diag(cnf_matrix)
TN = cnf_matrix.sum() - (FP + FN + TP)

FP = sum(FP)
FN = sum(FN)
TP = sum(TP)
TN = sum(TN)

# Sensitivity, hit rate, recall, or true positive rate
TPR = float(TP)/(TP+FN)
# Specificity or true negative rate
TNR = float(TN)/(TN+FP) 
# Precision or positive predictive value
PPV = float(TP)/(TP+FP)
# Negative predictive value
NPV = float(TN)/(TN+FN)
# Fall out or false positive rate
FPR = float(FP)/(FP+TN)
# False negative rate
FNR = float(FN)/(TP+FN)
# False discovery rate
FDR = float(FP)/(TP+FP)

# Overall accuracy
ACC = float(TP+TN)/(TP+FP+FN+TN)

print "True Positive: " + str(TP)
print "False Positive: " + str(FP)
print "True Negative: " + str(TN)
print "False Negative: " + str(FN)

print "True positive rate: " + str(TPR)
print "True negative rate: " + str(TNR)
print "Positive predictive value: " + str(PPV)
print "Negative predictive value: "+ str(NPV)
print "False positive rate: " + str(FPR)
print "False negative rate: " + str(FNR)
print "False discovery rate: " + str(FDR)
print "Overall accuracy: " + str(ACC)
