'''this is the  knn clustering script, done separately for each feature
'''

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os

def image_to_feature_vector(image, size=(32, 32)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
	return cv2.resize(image, size).flatten()


def extract_color_histogram(image, bins=(8, 8, 8)):
	# extract a 3D color histogram from the HSV color space using
	# the supplied number of `bins` per channel
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
		[0, 180, 0, 256, 0, 256])
	# handle normalizing the histogram if we are using OpenCV 2.4.X
	if imutils.is_cv2():
		hist = cv2.normalize(hist)
	# otherwise, perform "in place" normalization in OpenCV 3 (I
	# personally hate the way this is done
	else:
		cv2.normalize(hist, hist)
	# return the flattened histogram as the feature vector
	return hist.flatten()


# construct the argument parse and parse the arguments
# --dataset : This is the path to our input Kaggle Dogs vs. Cats dataset directory
# --neighbors : Here we can supply the number of nearest neighbors that are taken into account when classifying a given data point. default is one
# --jobs : Finding the nearest neighbor for a given image requires us to compute the distance from our input image to every other image in our dataset. This is clearly a O(N) operation that scales linearly. For larger datasets, this can become prohibitively slow. In order to speedup the process, we can distribute the computation of nearest neighbors across multiple processors/cores of our machine. Setting --jobs  to -1  ensures that all processors/cores are used to help speedup the classification process.
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-k", "--neighbors", type=int, default=1,
	help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1,
	help="# of jobs for k-NN distance (-1 uses all available cores)")
args = vars(ap.parse_args())



# grab the list of images that we'll be describing
print("[INFO] describing images...")
# grab path to all images
imagePaths = list(paths.list_images(args["dataset"]))
# initialize the raw pixel intensities matrix, the features matrix,
# and labels list
rawImages = []
features = []
labels = []
