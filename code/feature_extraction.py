"""
This file contains functions for manually extracting certain properties from pre-processed asparagus
pieces and the utility functions that are needed.
"""

# import area
import cv2
import matplotlib
import numpy as np
import scipy.stats as stats
import skimage.measure as measure
from scipy.ndimage import label, find_objects
from preprocessor import *
from utils import *

def get_length(img):
    '''Simple length extraction
    The length is measured from the highest white pixel to the lowest in the binarized image after rotation
    Args:
        img: the image
    Returns:
        length: the length in millimeters from highest to lowest point, under the assumption that one pixel
                corresponds to 4 pixels
    '''
    img = rotate_to_base(img)
    upper, lower = find_bounds(img)
    length = lower - upper
    # TODO: Umrechnungsfaktor von Pixel zu mm
    return length/4.2

def get_horizontal_slices(img, k):
    """
    Calculates the x-coordinates of the outline of the asparagus pieces, measured at k evenly
    spaced horizontal slicing points.

    Args:
        img = the preprocessed image
        k = the number of slices

    returns:
        the slice_points (y-coordinates)
        an np array([a1, a2],[b1,b2] ... ) where a1,a2 = x-coordinates of asparagus outline
    """
    # find upper and lower bound of asparagus piece
    upper, lower = find_bounds(img)
    # evenly distribute the slices between the bounds, but start a little lower than the head
    # and end a little earlier than the bottom
    slice_points = np.floor(np.linspace(upper+100, lower-20, k))
    # slice the image at the slice_points and return the left and right pixel
    def slice_img(img, sp):
        sp = int(sp)
        bin_img = binarize(img, 20)
        line = np.nonzero(bin_img[sp,:])
        left = line[0][0]
        right = line[0][-1]
        return left, right

    return slice_points, np.array([[left, right] for left, right in [slice_img(img, sp) for sp in slice_points]])

def curvature_score(img, k):
    """ Returns a score for the curvature of the aparagus piece.
        A perfectly straight aspargus yields a score of 0
        Args:
            img: the image
            k: number of horizontal slices
        Returns:
            std_err (float): standard error of linear regression through the slices
    """
    rows, horizontal_slices = get_horizontal_slices(img, k)
    centers = np.mean(horizontal_slices, axis=1)
    std_err = stats.linregress(rows, centers)[-1]
    return std_err


def get_width(img, k):
    '''Extract the width at k different rows

    Args:
        img: the image from which the width should be extracted
        k: number of rows in which the width should be extracted

    Returns:
        min and max width of the k different rows (# of pixels)
    '''
    # rotate the image
    img = rotate_to_base(img)
    # get the horizontal slices
    _, horizontal_slices = get_horizontal_slices(img, k)
    # calculate the difference between the points in each slice
    width = np.diff(horizontal_slices, axis=1)
    # TODO: Umrechnungsfaktor von Pixel zu mm
    return np.max(width)/4.2, np.min(width)/4.2



def check_purple(img, threshold_purple=6, ignore_pale=0.3):
    """ Checks if an asparagus piece is purple.
    Args:
        img:                A numpy array representing an RGB image where masked out pixels are black.
        threshold_purple:   If the histogram of color-hues (0-100) has a peak below this threshold
                            the piece is considered to be purple.
        ignore_pale:        Don't consider pixels with a saturation value below ignore_pale
    Returns:
        bool: A boolean that indicates wether the piece is purple or not.
        list: A list representing the histogram of hues with 100 bins.
    Examples:
        >>> fig, ax = plt.subplots(2,1,figsize=(14,10))

        >>> is_purple, hist_hue_purple = check_purple(image_of_purple_piece)
        >>> print(is_purple)
        >>> is_purple, hist_hue_white = check_purple(image_of_white_piece)
        >>> print(is_purple)
        >>> ax[0].plot(hist_hue_purple)
        >>> ax[0].plot(hist_hue_white)
        >>> ax[1].imshow([np.linspace(0, 1, 256)], aspect='auto', cmap=plt.get_cmap("hsv"))
    """
    hsv = matplotlib.colors.rgb_to_hsv(img)
    hue = hsv[:,:,0]
    sat = hsv[:,:,1]
    bins = np.linspace(0,1,101)

    #Mask out black values:
    mask = ~np.logical_and(np.logical_and(img[:,:,0]==0, img[:,:,1]==0),img[:,:,2]==0)
    mask = np.logical_and(mask,sat>ignore_pale)


    hist = np.histogram(hue[mask],bins=bins)[0]

    peak = np.argmax(hist)

    is_purple = False
    if peak < threshold_purple:
        is_purple = True

    return is_purple, hist

def rust_counter(img, lower=np.array([50,42,31]), upper=np.array([220,220,55]), max_count=30000):
    """ Counts the number of pixels that might be rusty.
    Args:
        img: image
        lower: lower bound for color range of rust
        upper: upper bound for color range of rust
        max_count: to normalize return value (return value around 0.13 is allready rusty)
    Returns:
        value: normalized to range from 0 to 1
    """
    # find the pixels that are in the range of rusty colors
    rust_mask = cv2.inRange(img, lower, upper)
    # put the preprocessed image and the rust_mask together to get rid of the background AND the bright pixels
    output = cv2.bitwise_and(img, img, mask = rust_mask)
    # count the remaining pixels
    count = np.count_nonzero(output)
    # normalize the count to the range of 0 to 1 to make it easier to interpret
    value = count/max_count
    # plot for debugging/to see whether the bounds are okay
    # fig = plt.figure()
    # ax1 = fig.add_subplot(1,2,1)
    # ax1.imshow(rust_mask)
    # ax2 = fig.add_subplot(1,2,2)
    # ax2.imshow(img)
    # fig.suptitle("rust count = " + str(value))
    # plt.show()

    return value

def zero_crossing(img):   
    """
    Use zero crossing to detect edges and display them.
    Filter the image with a laplacian to get an approximation of the second derivate and apply zero crossing.
    This function does not give a return value, but plots the results for visual inspection.
    Note: This function was an attempt for flower detection, but it was never finished.
    Args:
        img: the image
    Return:
        None
    """
    # convert image to grey scale
    img = color.rgb2gray(img)
    # smooth the image
    img_smoothed = filters.gaussian(img, sigma=4) # or 2.0, 4.0
    # detect edges using a laplacian filter
    edges = filters.laplace(img_smoothed)
    # N4 neighborhood
    zero_crossings_n4 = (edges[:-1, 1:] * edges[1:, 1:] <= 0) | (edges[1:, :-1] * edges[1:, 1:] <= 0)
    # N8 neighborhood
    zero_crossings_n8 = (zero_crossings_n4[:, 1:] 
                        | (edges[:-1, 1:-1] * edges[1:, :-2] <= 0) 
                        | (edges[:-1, 1:-1] * edges[1:, 2:] <= 0))
    # plot the results
    plt.figure(figsize=(12, 12))
    plt.gray()

    plt.subplot(2,2,1); plt.axis('off'); plt.imshow(img); plt.title('original')
    plt.subplot(2,2,2); plt.axis('off'); plt.imshow(edges); plt.title('edges')
    plt.subplot(2,2,3); plt.axis('off'); plt.imshow(zero_crossings_n4); plt.title('zero crossings (N4)')
    plt.subplot(2,2,4); plt.axis('off'); plt.imshow(zero_crossings_n8); plt.title('zero crossings (N8)' )

    plt.show()