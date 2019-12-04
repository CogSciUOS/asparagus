# load needed libraries
import os
import sys
import re
from PIL import Image
import sys
import traceback
import numpy as np
from scipy.ndimage.morphology import binary_hit_or_miss
from scipy.ndimage.measurements import center_of_mass
import csv
import matplotlib
# Import PCA Algorithm
from sklearn.decomposition import PCA


# here are the test images: Z:\net\projects\scratch\summer\valid_until_31_January_2020\asparagus\asparagus\images\raw_data


os.makedirs(outpath, exist_ok=True)#Make dir if non existant

# put 3 images of one asparagus piece into one list
imgs = []# open images
    try:
        imgs.append(Image.open(fpath1))
        imgs.append(Image.open(fpath2))
        imgs.append(Image.open(fpath3))
        assert len(imgs) == 3
        assert len(list(np.array(imgs[0]).shape)) == 3
        assert len(list(np.array(imgs[1]).shape)) == 3
        assert len(list(np.array(imgs[2]).shape)) == 3
    except Exception as e:
        print("Could not load all images correctly. Triple:")
        print(triple)
        print(e)
        return

    outpaths = [outpath+str(file_id)+"_a.png",outpath+str(file_id)+"_b.png",outpath+str(file_id)+"_c.png"]



# Initialize the algorithm and set the number of PC's
pca = PCA(n_components=2)

# Fit the model to data
pca.fit(data)
# Get list of PC's
pca.components_
# Transform the model to data
pca.transform(data)
# Get the eigenvalues
pca.explained_variance_ratio


if name == "__main__":
    # to start with the submit script: define arguments
    path_to_valid_names = sys.argv[1]#contains initfile (filenames)
    outpath = sys.argv[2]#must contain a slash at the end
    start_idx = int(sys.argv[3])
    stop_idx = int(sys.argv[4])

    #path_to_valid_names = "/net/projects/scratch/summer/valid_until_31_January_2020/asparagus/Images/unlabled/valid_files.csv"
    path_to_valid_names = /Desktop/Master/StudyProject/asparagus/images/raw_data
    # get valid file names
    valid_triples = []
    with open(path_to_valid_names, 'r') as f:
        reader = csv.reader(f)
        # only read in the non empty lists
        for row in filter(None, reader):
            valid_triples.append(row)



for triple in valid_triples[start_idx:stop_idx]:
        out = outpath

file_id += 1
