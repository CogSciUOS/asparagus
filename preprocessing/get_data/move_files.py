'''
This script takes all asparagus IDs from a csv file (in this case all IDs that we hand labeled) and moves the corresponding image files to the desired location.
'''

import pandas as pd
import numpy as np
import os
import sys
import shutil

grid_dir = (os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
+ '\\grid_framework')
sys.path.append(grid_dir)

from grid import*


def get_asparagus_ids(PATH):
    '''
    Get ids of the asparagus pieces that have been labeled so far.
    Args: path to the combined.csv file that contains all labelfiles
    Out: only the ids aka the first column
    '''
    # read in the file
    csvs = pd.read_csv(PATH, sep = ';')
    # the column corresponding to the ids
    ids = csvs['id']
    # make it a numpy array for better parsing 
    ids = np.array(ids)
    return ids

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
                file_names.append(int(file[:-6])) #modified to not save without _a.png and save as int for later comparison
    return all_files, file_names

if __name__ == '__main__':
    args = typecast(sys.argv[1:])
    
    path_to_imgs = args[0]
    path_to_csv = args[1]
    path_to_save = args[2]
    # read ids from combined.csv
    ids = get_asparagus_ids(path_to_csv)
    print('#ids: ' + str(len(ids)))
    # get all images file names and corresponding paths
    file_paths, file_names = get_files(path_to_imgs)
    print('#files found: ' + str(len(file_names)))
    files = np.array(file_names)
    print(files[0:10])
    index_list = []
    for item in ids:
        item_index = np.where(files==item)
        for idx in item_index[0]:
            shutil.copy(file_paths[int(idx)], path_to_save)
        # to check whether all were found count the number of found indices
        index_list.append(item_index)
    print('#indices found: ' + str(len(index_list)))
