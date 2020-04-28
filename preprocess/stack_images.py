'''
This script takes the three perspectives of one asparagus piece and stacks them. They can be stacked horizontally (side by side) or vertically (one after another).
'''
# load packages
from matplotlib.pyplot import imread
#import pandas as pd
import numpy as np
import os
from grid import*
import sys


def _mkdir(newdir):
    """works the way a good mkdir should :)
        - already exists, silently complete
        - regular file in the way, raise an exception
        - parent directory(ies) does not exist, make them as well
    """
    if os.path.isdir(newdir):
        pass
    elif os.path.isfile(newdir):
        raise OSError("a file with the same name as the desired " \
                      "dir, '%s', already exists." % newdir)
    else:
        head, tail = os.path.split(newdir)
        if head and not os.path.isdir(head):
            _mkdir(head)
        #print "_mkdir %s" % repr(newdir)
        if tail:
            os.mkdir(newdir)

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
    # load the images - this exceeds the memory, they have to be loaded one by one
    #images = [imread(f) for f in file_paths]
    # number of asparagus pieces
    n = int(len(file_paths)/3)
    print(n)
    # use a counter to save the images in subfolders - only for large amount of images
    count = 0
    idx = 0
    for i in range(0, len(file_paths), 3):
        # load the three corresponding images and make them a numpy array
        img_a = imread(file_paths[i])
        df_a = np.array(img_a)
        img_b = imread(file_paths[i+1])
        df_b = np.array(img_b)
        img_c = imread(file_paths[i+2])
        df_c = np.array(img_c)
        df_concat = np.concatenate((df_a, df_b, df_c), axis = 1) # change the axis here to stack in different direction
        # get the filename of the image to save the stacked image with the same number
        filename = file_names[i]
        # remove _a.png and add _stacked instead
        new_name = filename[:-6] + '_stacked_noB_h'
        folder = str(path_out + str(idx) + '/')
        # create the folder if it doesn't exist
        _mkdir(folder)
        save_to = folder + new_name
        #save_to = str(path_out + count + '/' + new_name + '_stacked')
        # save the stacked images
        np.save(save_to, df_concat)
        count += 1
        print(count)
        # create a new folder after 1000 images so the folders don't get to big
        #if count%1000 == 0:
        #    idx += 1
        

if __name__ == '__main__':

    args = typecast(sys.argv[1:])
    path_in = args[0]
    path_out = args[1]
    file_paths, file_names = get_files(path_in)
    stack_images(file_paths, file_names, path_out)
