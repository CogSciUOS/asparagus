import os
import sys
import math

import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import rotate, shift
from scipy.ndimage.morphology import binary_erosion, binary_opening, binary_closing

from skimage.measure import block_reduce

from skimage.transform import hough_line, hough_line_peaks

"""
A collection of utility classes for manipulating asparagus images
"""

def binarize(img, thres):
    """
    binarizes an image based on thresshold
    """
    res = np.sum(img,axis=2) > thres
    return res.astype(int)


def downsample(img):
    """
    quickly downsamples image through mean interpolation
    """
    res = block_reduce(img, block_size=(21, 21), func=np.mean)
    return res


def erode(img):
    """
    run vanilla erosion on image
    """
    res = binary_erosion(img, np.ones((3,3), dtype=np.int))
    return res


def find_angle(img):
    """
    Finds the angle of rotation of the asparagus piece in the image.
    returns that angle
    """
    img = binarize(img, 2)
    img = erode(img)
    h_space, angles, dist = hough_line(img)
    h_space, angles, dist = hough_line_peaks(h_space, angles, dist, num_peaks=1)
    return math.degrees(angles[0])


def rotate_to_base(img):
    """
    Finds out what the asparagus angle is, zeroes image by rotating it to 0°
    """
    angle = find_angle(img)
    base = rotate(img, angle, reshape=False, mode="constant")
    return base


def find_bounds(img):
    """
    Finds the upper and lower limit of the nonzero box, that is: 
    Where the asparagus piece starts and where it ends.
    
    returns: upper and lower, the limits of the asparagus piece

    WARNING: the image rows are numbered from top to bottom, so the first = highest
             row of the image is row 0 and the last = lowest is 1200.
             Therefore, upper is the limit at the head of the asparagus piece, lower
             is at the bottom of the asparagus piece. 
             BUT: this means that the value of lower is higher than the value of upper.
    """
    img = np.array(img)
    img = binarize(img, 20)
    
    collapse = np.sum(img, axis=1)
    nonz = np.nonzero(collapse)
    
    upper = nonz[0][0]
    lower = nonz[0][-1]

    return upper, lower

def head_finder(img):
    """Cut image to only display the head of the asparagus.
    Args:
        img: the image
    Return:
        head: the image cropped around the head
    """
    # binarize the image
    img_bin = binarize(img,20)
    # indices of non empty rows
    ind_row = np.nonzero(img_bin.any(axis=1))[0] 
    # the first one is the highest row with non-zero pixels
    upper = ind_row[0]
    # crop the image rows
    img_crop = img_bin[upper-20:upper+200,:]
    # indices of non empty columns 
    ind_col = np.nonzero(img_crop.any(axis=0))[0] 
    # find left most and right most column
    left = ind_col[0]
    right = ind_col[-1]
    # crop image columns
    head = img[upper-20:upper+200,left:right,:]

    return head