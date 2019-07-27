import numpy as np
import cv2
import matplotlib.pyplot as plt

def rust_counter(img, lower, upper, threshold):
    """ Counts the number of pixels that might be rusty.
    Args:
        img: image
        lower: lower bound for color range of rust
        upper: upper bound for color range of rust
        threshold: how many pixels have to be rusty before it's classified as rust?
    Returns:
        boolean: True if it's rusty, false otherwise
    """
    rust_mask = cv2.inRange(img, lower, upper)
    output = cv2.bitwise_and(img, img, mask = rust_mask)
    count = np.count_nonzero(output)
    if count > threshold:
        return True
    else:
        return False

if __name__ == "__main__":
    # read in the image
    img = plt.imread("C:/Users/Sophia/Documents/GitHub/asparagus/rust_selected/new_prepro/7_2.jpg")
    # set the lower und upper bound for "rusty-colours"
    # TODO: find good upper and lower bounds and threshold
    # maybe train the threshold on the labeled images?
    lower = np.array([50,42,31])
    upper = np.array([220,220,55])
    threshold = 2000
    rust = rust_counter(img, lower, upper, threshold)
    print(rust)