import cv2
import matplotlib
import numpy as np
import scipy.stats as stats 
import skimage.measure as measure
from scipy.ndimage import label, find_objects
from preprocessor import *
from utils import *
from scipy import misc
import matplotlib.pyplot as plt
from skimage import color, filters

def filter(img):   
    
    img = color.rgb2gray(img)

    # Now compute edges and then zero crossings using the 4-neighborhood and the 8-neighborhood
    # BEGIN SOLUTION

    # from scipy.ndimage.filters import laplace, gaussian_laplace
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
    # END SOLUTION

    plt.figure(figsize=(12, 12))
    plt.gray()

    plt.subplot(2,2,1); plt.axis('off'); plt.imshow(img); plt.title('original')
    plt.subplot(2,2,2); plt.axis('off'); plt.imshow(edges); plt.title('edges')
    plt.subplot(2,2,3); plt.axis('off'); plt.imshow(zero_crossings_n4); plt.title('zero crossings (N4)')
    plt.subplot(2,2,4); plt.axis('off'); plt.imshow(zero_crossings_n8); plt.title('zero crossings (N8)' )

    plt.show()
    return None
if __name__ == "__main__":

    import os
    directory = "C:/Users/Sophia/Documents/GitHub/asparagus/aussortiert_von_Blume/"
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"): 
            path = os.path.join(directory, filename)
            img = plt.imread(path)
            head = head_finder(img)
            head_gray = color.rgb2gray(head)
            filter(head_gray)
            continue
        else:
            continue

