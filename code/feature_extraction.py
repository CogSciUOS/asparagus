# created by Richard Ruppel at 2019/09/11
# last changes from ******** at 2019/09/**

# import area
import doctest
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
from scipy.ndimage.morphology import binary_hit_or_miss


def preprocess_violett(path):
    """ Initiates masking in the hsv space.

    Cuts out the area, where is not blue.

    Args:
        path (string): path to Image

    Returns:
        Image: processed Image 
    """

    # Open Image
    raw = np.array(Image.open(path))
    # remove alpha-channel (only if its RGBA)
    raw = raw[:,:,0:3]
    # transform to HSV color space
    hsv = matplotlib.colors.rgb_to_hsv(raw)
    
    # Mask all blue hues
    mask = np.logical_and(hsv[:,:,0]>0.4 , hsv[:,:,0]<0.8)
    # Mask out values that are not bright enough
    mask = np.logical_or(hsv[:,:,2]<100,mask)
    
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

    mask = np.logical_not(m)
    raw[:,:,0][mask] = 0
    raw[:,:,1][mask] = 0
    raw[:,:,2][mask] = 0
    return raw




# TODO: 
# - Return a meaningful value
# - change input
def get_violett(arg1):
    """Checks for violett parts in the picture.

    Extended description of function.

    Args:
        arg1 (int): Description of arg1

    Returns:
        bool: Description of return value
    """

    img = preprocess_violett(arg1) 
    # img = preprocess_violett(arg2) #np.array(Image.open(os.getcwd()))
    # img = preprocess_violett(os.getcwd()+"/npurp3.png") #np.array(Image.open(os.getcwd()))
    # img = preprocess_violett(np.array(Image.open(os.getcwd())))
       
    



    return True




def func(arg1, arg2):
    """Summary line.

    Extended description of function.

    Args:
        arg1 (int): Description of arg1
        arg2 (str): Description of arg2

    Returns:
        bool: Description of return value
    """
    return True








if __name__ == "__main__":

    # TODO: Move to executing function, make sure file is there or open filePicker
    # mask = preprocess_violett(os.getcwd() + "/npurp1.png")

    
    doctest.testmod()




