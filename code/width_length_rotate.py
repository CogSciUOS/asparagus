import numpy as np
#import matplotlib.pyplot as plt
from skimage import filters, io
from scipy.ndimage.morphology import binary_opening #, binary_closing, binary_dilation
from scipy.ndimage import label, find_objects
import skimage as sk
from scipy.ndimage.interpolation import rotate
from sklearn.decomposition.pca import PCA

# functions from Thomas' preprocessing
# I just copied these and altered them a little to fit my needs

def binarize_asparagus_img(img):
    
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
    
    img_bw = np.sum(img, axis=2)
    img_bw = img_bw/np.max(img_bw)*255
    white = img_bw > 90
    blue = blue_delta(img) > 25
    return np.logical_and(white, np.invert(blue))

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
    img_labeled = sk.measure.label(img_mask)
    # regionprops extracts all kinds of features from the labeld image
    props = sk.measure.regionprops(img_labeled)
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

if __name__ == "__main__":

    ### this should be in the main later
    # load the image
    img = io.imread('C:/Users/Sophia/Documents/GitHub/asparagus_alt/clean_images/13_2.jpg').astype(float)
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
