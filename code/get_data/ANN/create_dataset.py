import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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
        for file in files[:1]:
            filepath = subdir + '/' + file
            if filepath.endswith(".jpeg"):
                all_files.append(filepath)
    return all_files

# def load_images(PATH):
#     files = os.listdir()

        
#         img = plt.imread(PATH + )
#     return data

# def preprocess_img(img):
#     '''
#     resize, convert to Lab
#     '''
    
#     return processed_img


if __name__ == '__main__':
    #args = typecast(sys.argv[1:])

    PATH = 'Z:/net/projects/data/ImageNet/ILSVRC2012/train'
    files = get_files(PATH)
    print(files)

