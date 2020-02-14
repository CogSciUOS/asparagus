import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys

#import skimage.color as color
#import skimage.io as io

#from grid import*
#from submit_create import*


def get_files(PATH):
    '''
    Get all file names in directories and subdirectories.
    Args: PATH to files
    Out: List of all file names and the corresponding directories
    '''
    all_files = []
    for subdir, dirs, files in os.walk(PATH):
        for file in files[:1]: # take 1 image from each folder aka each category --> change this to 100 (or more) later
            filepath = subdir + '/' + file
            if filepath.endswith(".JPEG"):
                all_files.append(filepath)
    return all_files

def preprocess(filepath):
    rgb = cv2.imread(filepath)
    print(rgb.shape)
    rgb_crop = cv2.resize(rgb, dsize = (224,224,3), interpolation = cv2.INTER_AREA)
    lab = cv2.cvtColor(rgb_crop, code = CV_RGB2Lab)
    return lab


if __name__ == '__main__':
    #args = typecast(sys.argv[1:])

    PATH = '/net/projects/data/ImageNet/ILSVRC2012/train/'
    files = get_files(PATH)
    print(len(files))
    labs = []
    for file in files:
        labs.append(preprocess(file))
    print(len(labs))

