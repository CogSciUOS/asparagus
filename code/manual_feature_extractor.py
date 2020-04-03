from PIL import Image
import os
import doctest
from skimage import filters, io
import numpy as np
from preprocessor import *
from feature_extraction import *
from augmentation import *

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
    curvature = curvature_score(get_slices(img, k, min_row), get_horizontal_slices(img, k, min_row))

    print(curvature)


### Thomas' Main from preprocessor.py

    img_dir = "raw_data/"
    target_dir = "clean_images_demo/"

    try:
        with open(os.path.join(img_dir, "progress.txt")) as file:
            pass
    except IOError:
        file = open(os.path.join(img_dir, "progress.txt"), "w")
        file.close()

    try:
        with open(os.path.join(target_dir, "names.csv")) as file:
            pass
    except IOError:
        file = open(os.path.join(target_dir, "names.csv"), "a")
        file.close()

    avg_width = 160
    avg_height = 1050

    max_width = 250
    max_height = 1200

    preprocessor(img_dir, target_dir, show=False, save=True, debug=False)