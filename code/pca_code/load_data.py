
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.spatial.distance import cdist
import os
import sys
import shutil



eig_used = np.load('/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/data_width/eig_hollow_used.npy')

#all_pc = np.load('/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/data_width/PC_width.py')


print("dim aspa_space: \n",eig_used)

#print(all_pc)
