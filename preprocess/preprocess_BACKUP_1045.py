<<<<<<< HEAD
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

def preprocess(triple,outpath,file_id):
    fpath1,fpath2,fpath3 = triple

    os.makedirs(outpath, exist_ok=True)#Make dir if non existant


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

    width = 364#width of snippet
    x_centers = [40+width//2,380+width//2,725+width//2]#center locations of snippet

    lowest = 1340

    for original_img, x_center, out in zip(imgs, x_centers, outpaths):
        im = np.array(original_img)
        leftmost  = x_center-width//2
        rightmost = x_center+width//2
        im = im[:lowest,leftmost:rightmost]#crop
        im = remove_background(im)

        #Shift coordinates such that center of gravity is in middle
        # this if-else is to fix the bug when running this on the grid - might delete if turns out wrong
        if np.isnan(im[:,:,0]).any():
            continue
        else:
            y,x = np.array(center_of_mass(im[:,:,0]), dtype=np.int32)
            shift = x-width//2
            x_center = x_center + shift

            im = np.array(original_img)
            pad = (width//2)+1
            im = np.pad(im,[[0,0],[pad,pad],[0,0]], 'constant')
            x_center += pad
            im = im[:lowest,x_center-width//2:x_center+width//2]#crop
            im = remove_background(im)


        Image.fromarray(im).save(out)

def remove_background(img_array):
    raw = img_array.copy()
    raw = raw[:,:,0:3]
    hsv = matplotlib.colors.rgb_to_hsv(raw)

    mask = np.logical_and(hsv[:,:,0]>0.4 , hsv[:,:,0]<0.8) #Mask all blue hues
    mask = np.logical_or(hsv[:,:,2]<80,mask) #Mask out values that are not bright engough

    #Use binary hit and miss to remove potentially remaining isolated pixels:
    m = np.logical_not(mask)
    change = 1

    while(change > 0):
        a = binary_hit_or_miss(m, [[ 0, -1,  0]]) + binary_hit_or_miss(m, np.array([[ 0, -1,  0]]).T)
        m[a] = False
        change = np.sum(a)

    mask = np.logical_not(m)
    raw[:,:,0][mask] = 0
    raw[:,:,1][mask] = 0
    raw[:,:,2][mask] = 0
    return raw


if __name__ == "__main__":
    # to start with the submit script: define arguments
    path_to_valid_names = sys.argv[1]#contains initfile (filenames)
    outpath = sys.argv[2]#must contain a slash at the end
    start_idx = int(sys.argv[3])
    stop_idx = int(sys.argv[4])

    #path_to_valid_names = "/net/projects/scratch/summer/valid_until_31_January_2020/asparagus/Images/unlabled/valid_files.csv"
    # get valid file names
    valid_triples = []
    with open(path_to_valid_names, 'r') as f:
        reader = csv.reader(f)
        # only read in the non empty lists
        for row in filter(None, reader):
            valid_triples.append(row)


    # where to save the output
    # NOTE: remember to put / at the end for the subfolders
    #outpath = "/net/projects/scratch/summer/valid_until_31_January_2020/asparagus/Images/preprocessed/"
    #files.sort()

    file_id = start_idx
    files_per_folder = 10000

    for triple in valid_triples[start_idx:stop_idx]:
        out = outpath
        preprocess(triple,out,file_id)
        file_id += 1
=======
import os.path
import os
import csv
from get_valid_triples import get_valid_triples
from submit_script import submit_script
from preprocess_locally import *
import argparse

description = """
    Use for example:
    python preprocess.py --mode grid --root /net/projects/scratch/summer/valid_until_31_January_2020/asparagus/Images/unlabled/ 
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--mode', help='Run the program either in mode local or in mode grid', nargs="?")
    parser.add_argument('--root', help='This folder must contain .bmp files that match the naming convention. \n These files may reside in subfolders.)', nargs="?")
    parser.add_argument('--output_directory', help='Run the program either in mode local or in mode grid', nargs="?")

    mode = parser.parse_args().mode or 'local'
    root = parser.parse_args().root or "/net/projects/scratch/summer/valid_until_31_January_2020/asparagus/Images/labled/1A_Anna"
    outpath = parser.parse_args().output_directory or os.getcwd()+"/test"


    parser.add_argument('--ignore', help='Specify this string (ignore) if you want to ignore folders or files that contain ignore', nargs="?")
    ignore = parser.parse_args().ignore or "before2019"

    #safe list of valid names in a csv file; each row contains a triplet of file directories
    valid = get_valid_triples(root, ignore)
    with open(os.path.join(os.getcwd(), 'valid_files.csv'), 'w') as outfile:
        writer = csv.writer(outfile, delimiter=',')
        for i in valid:
            writer.writerow(i)

    if mode == "local":
        assert False
        perform_preprocessing(os.path.join(os.getcwd(),'valid_files.csv'), outpath, 0, -1)

    elif mode == "grid":
        #root = "/net/projects/scratch/summer/valid_until_31_January_2020/asparagus/Images/unlabled"
        #ignore = "before2019"

        n_pieces = len(valid)
        n_jobs = 5
        files_per_job = n_pieces // n_jobs
        for job in range(n_jobs):
            current_outpath = outpath + "/" + str(job)
            if job == n_jobs-1:
                args = [os.path.join(os.getcwd(),'valid_files.csv'), current_outpath, str(job*files_per_job), str(n_pieces)]
            else:
                args = [os.path.join(os.getcwd(),'valid_files.csv'), current_outpath, str(job*files_per_job), str(job*files_per_job+files_per_job-1)]
            submit_script(os.getcwd()+"/preprocess_locally.py",args)

        """

        args = [os.path.join(os.getcwd(),'valid_files.csv'), os.getcwd()+"/test", 0, -1]
        args = [str(x) for x in args]
        submit_script(os.path.join(os.getcwd(),"preprocess_locally"))"""

    #"/net/projects/scratch/summer/valid_until_31_January_2020/asparagus/Images/unlabled/"


>>>>>>> origin/variational_auto_encoder
