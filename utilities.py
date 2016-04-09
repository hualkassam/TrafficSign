__author__ = 'Hussam_Qassim'

import os # https://docs.python.org/2/library/os.html
import sys # https://docs.python.org/2/library/sys.html
import glob # https://docs.python.org/2/library/glob.html
import random # https://docs.python.org/2/library/random.html
import numpy as np
from time import time

from skimage.io import imread # http://scikit-image.org/docs/dev/api/skimage.io.html
from skimage.transform import resize # http://scikit-image.org/docs/dev/api/skimage.transform.html?highlight=skimage.transform#module-skimage.transform
from sklearn import preprocessing # http://scikit-learn.org/stable/modules/preprocessing.html
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score # http://scikit-learn.org/stable/modules/model_evaluation.html
from sklearn.cross_validation import StratifiedShuffleSplit # http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

sys.setrecursionlimit(1000000) # https://docs.python.org/2/library/sys.html

pixelNo = 128

# This path should be reset when changing computer
root = "/home/hualkassam/Desktop"


def readTrainingImages():
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



def scaleImages(images):
    avg = np.mean(images, axis=0)
    images -= avg
    std = np.std(images, axis=0)
    images /= std
    return images


def load_images():
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


def split_data(images, targets, random_state=0):
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


