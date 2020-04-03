# last changes from RR and SSW at 2019/09/12

# import area
import matplotlib
import numpy as np
import scipy.stats as stats 
import skimage.measure as measure
from scipy.ndimage import label, find_objects
# import from own scripts (have to be in the same folder!)
from preprocessor import binarize_asparagus_img, filter_mask_img, verticalize_img

def get_horizontal_slices(img, k, min_row):
    """Get the start and end values of a asparagus piece in a certain row.

    This is helper function for the curvature_score.

    Args:
        img : image
        k (int): Number of slices
        min_row (int): First row of the asparagus piece

    Returns:
        horizontal_slices: Pairs of values for each of the k rows
    """

    # binarize and filter the image with functions from preprocessor.py
    img_mask = filter_mask_img(binarize_asparagus_img(img))
    # get the slices 
    slices = get_slices(img, k, min_row)
    horizontal_slices = np.zeros((k,2))
    # find the first and the last 1-value in the row and save it in horizontal_slices
    for i in range(k):
        start = np.argwhere(img_mask[slices[i]]==True)[0]
        horizontal_slices[i][0] = start[0]
        end = np.argwhere(img_mask[slices[i]]==True)[-1]
        horizontal_slices[i][1] = end[0]
    
    return horizontal_slices


def curvature_score(slices, horizontal_slices):
    """ Returns a score for the curvature of the aparagus piece. 
        A perfectly straight aspargus yields a score of 0
        Args:
            slices (np.array): from get_slices function
            horizontal_slices (np.array): from get_horizontal_slices function
        Returns:
            std_err (float): standard error of linear regression through the slices
    """
    centers = np.mean(horizontal_slices, axis=1)
    _, _, _, _, std_err = stats.linregress(slices, centers)
    
    return std_err


def get_length(img):
    '''Simple length extraction
    The length is measured from the highest white pixel to the lowest in the binarized image after rotation
    Args:
        img: the image
    Returns:
        length: the length from highest to lowest white pixel
        min_row: the row of the highest pixel
    '''
    # rotate the image so that it is upright
    img_rotated = verticalize_img(img)
    # use Thomas helper functions to get the boolean image
    img_mask = filter_mask_img(binarize_asparagus_img(img_rotated))
    # set labels to the different areas, which are in our case only two - background and asparagus
    img_labeled = measure.label(img_mask)
    # regionprops extracts all kinds of features from the labeld image
    props = measure.regionprops(img_labeled)
    # we only need the properties from the bounding box
    min_row, _, max_row, _ = props[0].bbox
    # finally we can calculate the length by subtracting the min from the max pixel position
    length = max_row - min_row
    
    return length, min_row
    
# helper function for width_extraction and horizontal_slices
def get_slices(img, k, min_row):
    '''Get rows to measure the width in
    Slice the image into k even parts in which the widths should be measured
    Args:
        img: image
        k: number of widths we want to measure
        min_row: row at which the asparagus piece starts
    Returns:
        slices: the k rows
    '''
    slices = []
    # get the length of the asperagus piece
    length, min_row = get_length(img)
    # calculate the distance between each slice
    slice_dist = int(length/k)
    # the first slice shouldn't be at the very top of the piece but a little further down
    start = min_row + int(slice_dist/2)
    # save the row where the distance should be measured in the array slices
    for i in range(k):
        row = start + i*slice_dist
        slices.append(row)
    return slices

def get_width(img, k, min_row):
    '''Extract the width at k different rows

    Args:
        img: the image from which the width should be extracted
        k: number of rows in which the width should be extracted
        min_row: highest row for extraction (we don't want to start at the head)

    Returns: 
        min and max width of the k different rows (# of pixels)
    '''
    # get the rows where to measure the width
    slices = get_slices(img, k, min_row)
    # preprocess the image
    img_mask = filter_mask_img(binarize_asparagus_img(img))
    width = np.zeros((k))
    # sum over the 1-values of the preprocessed image in the specific rows
    for i in range(k):
        width[i] = np.sum(img_mask[slices[i]])
    
    return np.max(width), np.min(width)



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
