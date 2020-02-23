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
    for i in matrix:
        if np.any(np.iscomplex(matrix)) == True:
            print('complex an der stelle: \n', matrix[i,:])
    #np.iscomplex(matrix)
    #print(np.iscomplex(matrix))
