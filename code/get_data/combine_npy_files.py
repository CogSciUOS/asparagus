import numpy as np
import os
from grid import*
import sys

def combine(PATH):
    '''
    Combine the stacked npy files into one data set and downsample them by only taking every 6th pixel.
    We used every 6th pixel because it seems like a downscale that is still good enough to see most details.
    One could also downscale more or less or with a different technique.

    Args: Path to stacked npy files
    Out: None (just save the dataset)
    '''
    all_files = os.listdir(PATH)
    n = len(all_files)
    # load first image to get dimensionality and dtype dynamically
    first = np.load(PATH + all_files[0])[::3,::3] #change here to downscale differently
    dtype = first.dtype
    l, w, d = first.shape
    # make some space for the dataset
    data = np.empty((n, l, w, d), dtype=dtype)
    # load all files and save them in the corresponding position in the data array
    for i,file in enumerate(all_files):
        data[i,:,:,:] = np.load(PATH + file)[::3,::3] #change here to downscale differently
        # print how far along we are
        if i%500==0:
            print(i)
    # save dataset
    path_out = PATH + "data_horizontal.npy"
    np.save(path_out, data)

if __name__ == '__main__':
    args = typecast(sys.argv[1:])
    path = args[0]
    combine(path)