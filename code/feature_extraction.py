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
from scipy.ndimage.morphology import binary_hit_or_miss


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

    plt.figure()
    plt.imshow(m)


    change = 1
    while(change > 0):
        a = binary_hit_or_miss(m, [[ 0, -1,  0]]) + binary_hit_or_miss(m, np.array([[ 0, -1,  0]]).T)
        m[a] = False
        change = np.sum(a)
        # plt.imshow(a) # printf(changes)
    
    plt.figure()
    plt.imshow(m)
    plt.show()

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

    # ugly but testing 
    # @Katha how to write test functions? 
    raw = np.array(Image.open(os.getcwd() + "/npurp1.png"))
    plt.figure("main raw")
    plt.imshow(raw)

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
    
    doctest.testmod()