# created by Richard Ruppel and Sophia Schulze-Weddige at 2019/09/11
# last changes from RR and SSW at 2019/09/12

# import area
import doctest
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
from scipy.ndimage import label, find_objects
from scipy.ndimage.morphology import binary_hit_or_miss, binary_opening
from scipy.ndimage.interpolation import rotate
import scipy.stats as stats 
import skimage.measure as measure
from skimage import filters, io
from sklearn.decomposition.pca import PCA






# functions from Thomas' preprocessing
# I (Sophia) just copied these and altered them a little to fit my needs
def binarize_asparagus_img(img):
    
    img_bw = np.sum(img, axis=2)
    img_bw = img_bw/np.max(img_bw)*255
    white = img_bw > 90
    blue = blue_delta(img) > 25
    
    return np.logical_and(white, np.invert(blue))



def get_horizontal_slices(img, k, min_row):
    """Summary line.

    Extended description of function.

    Args:
        arg1 (int): Description of arg1
        arg2 (str): Description of arg2

    Returns:
        bool: Description of return value
    """

    img_mask = filter_mask_img(binarize_asparagus_img(img))
    slices = get_slices(img,k, min_row)
    horizontal_slices = np.zeros((k,2))
    for i in range(k):
        start = np.argwhere(img_mask[slices[i]]==True)[0]
        horizontal_slices[i][0] = start[0]
        end = np.argwhere(img_mask[slices[i]]==True)[-1]
        horizontal_slices[i][1] = end[0]
    
    return horizontal_slices


def curvature_score(preprocessed_image, slices, horizontal_slices):
    """ Returns a score for the curvature of the aparagus piece. 
        A perfectly straight aspargus yields a score of 0
    """



    #print(slices)
    centers = np.mean(horizontal_slices, axis=1)
    #plt.scatter(slices, centers)   
    slope, intercept, r_value, p_value, std_err = stats.linregress(slices,centers)
    return std_err

# @Thomas, why were those nested?
def blue_delta(img):
    """Delta value of image
    
    Returns the delta between blue and the avg other channels
    Args: 
        img: image
    Returns:
        delta: between blue and the avg other channels
    """
    other = np.mean(img[:,:,0:2], axis=2)
    return img[:,:,2]-other
    
def filter_mask_img(img):
    """Opening on the binarized image.
    Args:
        img: image
    Returns:
        rotated_img: image after opening
    """
    # I only use opening here because I want to change the size of the asparagus piece as little as possible
    # I only want to remove small white spots that are most likely due to reflections
    img = binary_opening(img, structure=np.ones((21,21)))
    return img


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
    min_row, min_col, max_row, max_col = props[0].bbox
    # finally we can calculate the length by subtracting the min from the max pixel position
    length = max_row - min_row
    
    return length, min_row
    
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

def verticalize_img(img):
    """Rotate an image based on its principal axis.
    Args:
        img: image to be rotated
    Returns:
        rotated_img: image after rotation
    """
    # mask the image because we need a binary image (or other two dimensional array-like object) for this function
    img_mask = filter_mask_img(binarize_asparagus_img(img)) 
    # Get the coordinates of the points of interest
    X = np.array(np.where(img_mask > 0)).T
    # Perform a PCA and compute the angle of the first principal axes
    pca = PCA(n_components=2).fit(X)
    angle = np.arctan2(*pca.components_[0])
    # Rotate the image by the computed angle
    # Note we use the masked image to find the angle but rotate the original image now
    rotated_img = rotate(img, angle/np.pi*180+90)
    return rotated_img


def cut_background(img, background_max_hue, background_min_hue, background_brightness):
    """ Initiates masking in the hsv space.

    Cuts out background with specific hue values.

    Args:
        img (Image): Image as numpy array
        background_max_hue (float): [0-255] 
        background_min_hue (float): [0-255] 
        background_brightness (float): [0-255] 

    Returns:
        Image: Image without background 
    """
    
    # Open Image
    raw = img
    # remove alpha-channel (only if its RGBA)
    raw = raw[:,:,0:3]
    # transform to HSV color space
    hsv = matplotlib.colors.rgb_to_hsv(raw)
    
    # Mask all blue hues (background)
    mask = np.logical_and(hsv[:,:,0] > background_min_hue , hsv[:,:,0] < background_max_hue)
    # Mask out values that are not bright enough
    mask = np.logical_or(hsv[:,:,2]< background_brightness, mask)
    
    #Use binary hit and miss to remove potentially remaining isolated pixels:
    m = np.logical_not(mask)

    change = 1
    while(change > 0):
        a = binary_hit_or_miss(m, [[ 0, -1,  0]]) + binary_hit_or_miss(m, np.array([[ 0, -1,  0]]).T)
        m[a] = False
        change = np.sum(a)
        # plt.imshow(a) # printf(changes)
    
    mask = np.logical_not(m)
    raw[:,:,0][mask] = 0
    raw[:,:,1][mask] = 0
    raw[:,:,2][mask] = 0


    return raw




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


# just for testing
if __name__ == "__main__":

### TEST VIOLETT ################################################


    # ugly but testing 
    # @Katha how to write test functions? 
    raw = np.array(Image.open(os.getcwd() + "/npurp1.png"))
    
    # fix values:
    background_max_hue = 0.8
    background_min_hue = 0.4
    background_brightness = 100.0
    img = cut_background(raw, background_max_hue , background_min_hue, background_brightness)
    
    # fix values:
    max_sat = 0.3
    violett = 0
    violett = get_violett(img, max_sat)

    print("violett: ", violett)
    
### TEST LENGTH #################################################

    # load the image
    path = "./../images/clean_images/13_2.jpg"

    img = io.imread(path).astype(float)
    # set the pixel values to the right range
    img /= img.max()
    # how many rows do we want
    k = 5
    # get length
    length, min_row = get_length(img)
    # get width
    max_width, min_width = get_width(img, k, min_row)

    print("Maximal Width: ", max_width)
    print("Minimal Width: ", min_width)
    print("Length: ", length)

    
    doctest.testmod()
    
### TEST CURVATURE ###############################################
    img_mask = filter_mask_img(binarize_asparagus_img(img))
    curvature = curvature_score(img_mask, get_slices(img, k, min_row), get_horizontal_slices(img, k, min_row))

    print(curvature)