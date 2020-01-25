import numpy as np
import os
from grid import*
import sys

def combine(PATH):
    all_files = os.listdir(PATH)
    n = len(all_files)
    data = np.empty(n, 1340, 364, 9)
    for i,file in enumerate(all_files):
        data[i,:,:,:] = np.load(file)[::6,::6]
    path_out = PATH + "data.npy"
    np.save(path_out, data)

if __name__ == '__main__':
    args = typecast(sys.argv[1:])
    path = args[0]
    combine(path)