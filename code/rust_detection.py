import numpy as np
import cv2
import matplotlib.pyplot as plt

def rust_counter(img, lower, upper, max_count):
    """ Counts the number of pixels that might be rusty.
    Args:
        img: image
        lower: lower bound for color range of rust
        upper: upper bound for color range of rust
        max_count: to normalize return value (return value around 0.13 is allready rusty)
    Returns:
        value: normalized to range from 0 to 1
    """
    rust_mask = cv2.inRange(img, lower, upper)
    output = cv2.bitwise_and(img, img, mask = rust_mask)
    count = np.count_nonzero(output)
    # normalize the count to the range of 0 to 1 to make it easier to interpret
    value = count/max_count
    # plot for debugging
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(rust_mask)
    ax2 = fig.add_subplot(1,2,2)
    ax2.imshow(img)
    fig.suptitle("rust count = " + str(value))
    plt.show()
    
    return value






if __name__ == "__main__":
    # read in the image
    img = plt.imread("C:/Users/Sophia/Documents/GitHub/asparagus/rust_selected/new_prepro/7_1.jpg")
    
    # # set the maximal number of pixels that can be rusty to normalize the output to the range of 0 to 1
    # max_count = 30000
    # # set the lower und upper bound for "rusty-colours"
    # # these bounds are just from experimenting around so far
    # lower = np.array([50,42,31])
    # upper = np.array([220,220,55])
    # rust = rust_counter(img, lower, upper, max_count)
    # print(rust)

    from skimage.feature import shape_index
    from utils import *

    index = shape_index(binarize(img,20))
    plt.imshow(index)
    plt.show()
