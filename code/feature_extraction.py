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

def curvature_score(img):
    """ Returns a score for the curvature of the aparagus piece. 
        A perfectly straight aspargus yields a score of 0
        Args:
            slices (np.array): from get_slices function
            horizontal_slices (np.array): from get_horizontal_slices function
        Returns:
            std_err (float): standard error of linear regression through the slices
    """
    rows, horizontal_slices = get_horizontal_slices(img, 5)
    centers = np.mean(horizontal_slices, axis=1)
    _, _, _, _, std_err = stats.linregress(rows, centers)
    
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



# TODO: 
# - Return a meaningful value
# - change input
def get_violett(img, _max_set):
    """Checks for violett parts in the picture.

    Calculates a value of how violett part in ???

    Args:
        filePath (string): Path to image

    Returns:
        TODO: value: Description of return value
    """


    hue = matplotlib.colors.rgb_to_hsv(img)[:,:,0]
    sat = matplotlib.colors.rgb_to_hsv(img)[:,:,1]

    mask = ~np.logical_and(np.logical_and(img[:,:,0] == 0, img[:,:,1] == 0), img[:,:,2] == 0) # sobald eins true ist ist alles true
    
    # TODO: discussible 
    # set the pixel with low saturation to black
    mask = np.logical_and(mask, sat > _max_set) 

    #TODO:
    # calcualte return value ( what makes sense? #Pixel above threshold? hue and or satuation values above threshold?)
    # some normalized value [0 1]? 
    # or boolean?
    return_value = 0

    return return_value

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
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(rust_mask)
    ax2 = fig.add_subplot(1,2,2)
    ax2.imshow(img)
    fig.suptitle("rust count = " + str(value))
    plt.show()
    
    return value

