__author__ = 'Hussam_Qassim'

import os # This module provides a portable way of using operating system dependent functionality
import sys # This module provides access to some variables used or maintained by the interpreter and to functions that interact strongly with the interpreter
import glob # The glob module finds all the pathnames matching a specified pattern according to the rules used by the Unix shell, although results are returned in arbitrary order
import random # This module implements pseudo-random number generators for various distributions
import numpy as np
from time import time # This module implements the time

from skimage.io import imread # Utilities to read and write images in various formats
from skimage.transform import resize # Resize image to match a certain size
from sklearn import preprocessing # This package provides several common utility functions and transformer classes to change raw feature vectors into a representation that is more suitable for the downstream estimators.
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.cross_validation import StratifiedShuffleSplit # Provides train/test indices to split data in train test sets

sys.setrecursionlimit(1000000) # Set the maximum depth of the Python interpreter stack to limit. This limit prevents infinite recursion from causing an overflow of the C stack and crashing Python

pixelNo = 128

# This path should be reset when changing computer
root = "/home/hualkassam/Desktop"


def readTrainingImages(): # Read the training images
    imagesPath = root
    directory_names = list(set(glob.glob(os.path.join(imagesPath, "Dataset", "*"))\
        ).difference(set(glob.glob(os.path.join(imagesPath, "Dataset", "*.*")))))
    images  = list()
    targets = list()
    label   = 0
    for folder in directory_names:
        header = folder.split("/")[-1]
        #print header
        for fileNameDir in os.walk(folder):
            files_number = len(fileNameDir[2])
            if files_number < 1000:
			    fileNamedirectories = fileNameDir[2]
            elif (files_number > 1000):
                fileNamedirectories = fileNameDir[2][0:1000]
            else:
			    fileNamedirectories = fileNameDir[2]

            for fileName in fileNamedirectories:
                if fileName[-4:] != ".jpg":
                    continue
                imageReading = "{0}{1}{2}".format(fileNameDir[0], os.sep, fileName)
                images.append(resize(imread(imageReading, as_grey=True), (pixelNo, pixelNo)))
                targets.append(label)
            label += 1
    print "Labels: ", label, " Number of Images: ", len(targets)
    return images, targets



def scaleImages(images): # Scale the images
    avg = np.mean(images, axis=0)
    images -= avg
    std = np.std(images, axis=0)
    images /= std
    return images


def load_images(): # Load the images to the system
    print "Image Reading.."
    t = time()
    images, targets = readTrainingImages()
    print "Time: ", round(time()-t, 3), "Sec" "\n"

    images = np.asarray(images)
    targets = np.asarray(targets)

    print "Image Scaling.."
    t = time()
    images= scaleImages(images)
    print "Time: ", round(time()-t, 3), "Sec" "\n"

    return images, targets


def split_data(images, targets, random_state=0): # Split the dataset into Training (70%), Validation (15%) and Testing (15%)
    folds = StratifiedShuffleSplit(targets, n_iter=1, test_size=0.30, random_state=random_state)
    for train, test in folds:
        x_train, y_train = images[train], targets[train]
        x, y = images[test], targets[test]

    folds = StratifiedShuffleSplit(y, n_iter=1, test_size=0.50, random_state=random_state)
    for valid, test in folds:
        x_valid, y_valid = x[valid], y[valid]
        x_test, y_test = x[test], y[test]

    print ('Training:%s') %len(x_train), "" ,('Validation:%s') %len(x_valid), "", ('Testing:%s') %len(x_test)

    x_train = x_train.reshape((-1, 1, pixelNo, pixelNo)).astype(np.float32)
    x_valid = x_valid.reshape((-1, 1, pixelNo, pixelNo)).astype(np.float32)
    x_test  = x_test.reshape((-1, 1, pixelNo, pixelNo)).astype(np.float32)

    y_train = y_train.astype(np.uint8)
    y_valid = y_valid.astype(np.uint8)
    y_test  = y_test.astype(np.uint8)

    return x_train, y_train, x_valid, y_valid, x_test, y_test


