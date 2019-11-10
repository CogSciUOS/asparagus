import os
import sys
import re
from PIL import Image
import sys
import traceback
import numpy as np
from scipy.ndimage.morphology import binary_hit_or_miss
from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage import label, find_objects

import csv
import matplotlib


iteration = 0
def preprocess(triple, outpath, file_id, outfiletype = "png", with_background = False):
    fpath1,fpath2,fpath3 = triple
    os.makedirs(outpath, exist_ok=True) #Make dir if non existant

    imgs = [] # open images
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
        print(e)
        return

    outpaths = [outpath+"/"+str(file_id)+"_a." + outfiletype,
                outpath+"/"+str(file_id)+"_b." + outfiletype,
                outpath+"/"+str(file_id)+"_c." + outfiletype]

    width = 364 #width of snippet
    x_centers = [40+width//2,380+width//2,725+width//2] #center locations of snippet

    lowest = 1340

    for original_img, x_center, out in zip(imgs, x_centers, outpaths):
        im = np.array(original_img)
        im = im[:,:,0:3] #Discard potentially existing transparancy value
        leftmost  = x_center-width//2
        rightmost = x_center+width//2
        im = im[:lowest,leftmost:rightmost]#crop
        im = remove_background(im)

        #Shift coordinates such that center of gravity is in middle
        #if np.isnan(im[:,:,0]).any():
        #    print("FALTAL ERROR")
        #    print(" The file with the index " + str(file_id)+" could not be generated")
        #    continue

        c = center_of_mass(im[:,:,0])
        c = np.nan_to_num(c)
        y,x = np.array(c, dtype=np.int32)
        shift = x-width//2
        x_center = x_center + shift

        im = np.array(original_img)
        pad = (width//2)+1
        im = np.pad(im,[[0,0],[pad,pad],[0,0]], 'constant')
        x_center += pad
        im = im[:lowest,x_center-width//2:x_center+width//2] #crop with new centers

        if not with_background:
            im = remove_background(im)
            im = remove_smaller_objects(im)

        Image.fromarray(im).save(out)

        # save the image file path and the file id in a csv file
        # NOTE: set this to True if needed
        save_path = False
        if save_path:
            # this automatically creates a file if it doesn't exist and else it appends to it
            with open('/net/projects/scratch/summer/valid_until_31_January_2020/asparagus/Images/img_dict.csv', 'a', newline = '') as fd:
                fd.write([triple, outpath, file_id])

        global iteration
        iteration += 1
        if iteration % 300 == 0:
            print("Created " + str(iteration) + " images", flush=True)

def remove_smaller_objects(image):
    image = image.copy()
    mask = image[:,:,0] != 0 #All foreground pixels are True
    labeled_image, num_features = label(mask) #Assign 1,2 ... to each group of connected pixels (that are True)
    objects = list(find_objects(labeled_image))
    sizes = [(widths.stop - widths.start)*(heights.stop - heights.start) for widths, heights in objects]
    objects = [x for _,x in sorted(zip(sizes,objects))] #Sort according to sizes
    for s in objects[:-1]: #Remove all but the object largest in size (of the bounding box)
        mask[s[0].start:s[0].stop,s[1].start:s[1].stop] = False

    mask = np.logical_not(mask)
    image[:,:,0][mask] = 0
    image[:,:,1][mask] = 0
    image[:,:,2][mask] = 0
    return image

def remove_background(img_array):
    raw = img_array.copy()
    hsv = matplotlib.colors.rgb_to_hsv(raw)

    mask = np.logical_and(hsv[:,:,0]>0.4 , hsv[:,:,0]<0.8) #Mask all blue hues
    mask = np.logical_or(hsv[:,:,2]<80,mask) #Mask out values that are not bright engough

    raw[:,:,0][mask] = 0
    raw[:,:,1][mask] = 0
    raw[:,:,2][mask] = 0
    return raw


def perform_preprocessing(initfile, outpath, startIdx, stopIdx, outfiletype, with_background):
    #root = "/net/projects/scratch/summer/valid_until_31_January_2020/asparagus/Images/unlabled/"
    # get valid file names
    try:
       valid_triples = []
       with open(initfile, 'r') as f:
          reader = csv.reader(f)
          # only read in the non empty lists
          for row in filter(None, reader):
              valid_triples.append(row)
    except Exception as e:
        print("Couldn't load initfile")
        print(e)
    print("Processing " + str(stopIdx-startIdx) + " triples")

    current_outfolder = 0
    triples_per_folder = 1000

    if stopIdx == -1:
        stopIdx = len(valid_triples)

    for idx, triple in zip(range(startIdx,stopIdx+1), valid_triples[startIdx:stopIdx+1]):
        if idx-startIdx % triples_per_folder == 0 and current_outfolder !=0:
            current_outfolder +=1

        out = outpath + "/" + str(current_outfolder)
        preprocess(triple,out,idx, outfiletype, with_background = with_background)


def print_usage():
    print("Provide the following arguments in the specified order: ")
    print("initfile: The path to your valid_files.csv")
    print("outpath: The path the preprocessed images shall be saved in")
    print("startIdx: The lower bound of files specified in your valid_files.csv that shall be processed")
    print("stopIdx: The upper bound of files specified in your valid_files.csv that shall be processed")

if __name__ == "__main__": # to start with the submit script: define arguments
    print("Gridjob started successfully... ")
    try:
        initfile = sys.argv[1]
        outpath = sys.argv[2]
        startIdx = sys.argv[3]
        stopIdx = sys.argv[4]
        outfiletype = sys.argv[5]
        with_background = sys.argv[6]
        try:
            startIdx = int(startIdx)
            stopIdx = int(stopIdx)
            with_background = int(with_background)
        except:
            print("startIdx and stopIdx (arg2) and arg(3) must be integers")
            print_usage()
    except:
        print("You did not provide a sufficient number of arguments")
        print_usage()
    perform_preprocessing(initfile, outpath, startIdx, stopIdx, outfiletype, with_background)
