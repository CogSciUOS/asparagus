# load packages
from matplotlib.pyplot import imread
#import pandas as pd
import numpy as np
import os
from grid import*
import sys

def get_files(PATH):
    '''
    Get all file names in directories and subdirectories.
    Args: PATH to files
    Out: List of all file names and the corresponding directories
    '''
    all_files = []
    file_names = []
    for subdir, dirs, files in os.walk(PATH):
        for file in files:
            filepath = subdir + '/' + file
            if filepath.endswith(".png"):
                all_files.append(filepath)
                file_names.append(file)
    return all_files, file_names


def stack_images(file_paths, file_names, path_out):
    '''
    Load images and stack the three images of the same asparagus after another.
    Save the image stack into a new folder. 
    Args: image file names
          path where to save the images
          the original file names 
    Out: None
    '''
    # load the images
    images = [imread(f) for f in file_paths]
    # number of asparagus pieces
    n = int(len(images)/3)
    print(n)
    # this counter is only to see whether the grid job is running
    count = 0
    for i in range(0, len(images), 3):
        # load the three corresponding images and make them a numpy array
        img_a = images[i]
        df_a = np.array(img_a)
        img_b = images[i+1]
        df_b = np.array(img_b)
        img_c = images[i+2]
        df_c = np.array(img_c)
        df_concat = np.concatenate((df_a, df_b, df_c), axis = 2)
        # get the filename of the image to save the stacked image with the same number
        filename = file_names[i]
        # remove _a.png
        new_name = filename[:-6]
        save_to = str(path_out + new_name + '_stacked')
        # save the stacked images
        np.save(save_to, df_concat)
        if count%1000 == 0:
            print(save_to)
        count += 1

if __name__ == '__main__':
    #path_in = 'C:/Users/Sophia/Documents/asparagus/code/variational_auto_encoder/images'
    #path_out = 'C:/Users/Sophia/Documents/asparagus/code/variational_auto_encoder/images/out/'
    args = typecast(sys.argv[1:])
    #path_in = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/with_background_pngs'
    #path_out = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/stacked_images/'
    path_in = args[0]
    path_out = args[1]
    file_paths, file_names = get_files(path_in)
    stack_images(file_paths, file_names, path_out)
