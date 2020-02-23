import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import sys
import shutil
from grid import*
from submit_test import*


if __name__ == '__main__':
    args = typecast(sys.argv[1:])
    matrix = np.load(args[0])
    #print(matrix[399,:])
    print(matrix.shape)
    #print(matrix[0,:])
    print(np.isrealobj(matrix))
    for x in matrix:
        if any(iscomplex(x, float) for x in matrix) == True:
            print(matrix[x])
    print((all(isinstance(x, float) for x in matrix)))
    #np.iscomplex(matrix)
    #print(np.iscomplex(matrix))
