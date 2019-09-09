import os
import sys
import re
from PIL import Image
import pickle
import sys
import traceback
import numpy as np
import matplotlib
from scipy.ndimage.morphology import binary_hit_or_miss
from scipy.ndimage.measurements import center_of_mass
import matplotlib.pyplot as plt

files = []
def get_valid_triples(root):
    
    """ List the files from which all three images exist.
    
    Unfortunately not all images exist, there are some single images. We don't use them for now
    but keep it in mind if we need more images later.
    
    Args: root: root directory of images
    Return: valid_triples: list of valid file names
    """
    # get the names of all files in the root directory and all subdirectories
    files = get_files(root,".bmp","before2019")

    valid_triples = [] 
    missing = []     
    # iterate over all file names
    for i,f in enumerate(files):
        triple = []
        # check whether first image of new asparagus
        if f.endswith("F00.bmp"):
            # get second and third image (same prefix, but ends with F01 and F02)
            second_perspective = f[:-7]+"F01.bmp"
            third_perspective = f[:-7]+"F02.bmp"
            # if those other two images exist append all to the valid_triples list
            if os.path.isfile(root+second_perspective) and os.path.isfile(root+third_perspective):
                triple.append(root+f)
                triple.append(root+second_perspective)
                triple.append(root+third_perspective)
                valid_triples.append(triple)
            else:
                continue
    return valid_triples

def rek_get_files(path, name_contains, ignore, root=None):
    for f in os.scandir(path):
        if ignore in f.path:
            continue
        if f.is_dir():
            print("Get all filenames in ... " + f.path)
            rek_get_files(f.path+"/", name_contains, ignore, root)
        else:
            if name_contains in f.name:
                if root == None:
                     files.append(path+f.name)
                else:
                     files.append((path+f.name)[len(root):])

def get_files(path, name_contains, ignore, use_full_path=False):
    files.clear()
    if use_full_path:
        rek_get_files(path, name_contains, ignore)
    else:
        rek_get_files(path, name_contains, ignore, root=path)
    return files

def preprocess(triple,outpath,file_id):    
    # controls if the programm still runs - can be deleted after testing
    print(".",end="")
    sys.stdout.flush()
    
    # make output directory if non existant for each preprocessor type (with and without background)
    prep_with = outpath + "prep_with/"
    prep_without = outpath + "prep_without/"
    os.makedirs(prep_with, exist_ok=True)
    os.makedirs(prep_without, exist_ok=True)

    # open the three images of the same asparagus piece
    imgs = [] 
    imgs.append(Image.open(triple[0]))
    imgs.append(Image.open(triple[1]))
    imgs.append(Image.open(triple[2]))
    
    # save the location of the image after preprocessing with and without background
    # naming convention: asparagusID_imageNumber
    #                    a = first, b = second, c = third image of asparagus piece
    outpaths_with = [prep_with+str(file_id)+"_a.png",prep_with+str(file_id)+"_b.png",prep_with+str(file_id)+"_c.png"]
    outpaths_without = [prep_without+str(file_id)+"_a.png",prep_without+str(file_id)+"_b.png",prep_without+str(file_id)+"_c.png"]    
    
    width = 364 # width of snippet
    
    #x_centers = [40+width//2, 380+width//2, 725+width//2] #center locations of snippet
    # guessed center of the three asparagus
    x_centers = [200, 500, 800]
    lowest = 1340

    for original_img, x_center, out_with, out_without in zip(imgs, x_centers, outpaths_with, outpaths_without):
        image = np.array(original_img)
        leftmost  = x_center-width//2
        rightmost = x_center+width//2
        # crop image
        im_with = image[:lowest,leftmost:rightmost]
        # remove background
        im_without = remove_background(im_with)
        plt.imshow(im_without)
        plt.show()
        # Shift coordinates such that center of gravity is in middle        
        y,x = np.array(center_of_mass(im_without[:,:,0]),dtype=np.int32)
        shift = x-width//2
        new_center = x_center + shift
        pad = (width//2)+1
        img_pad = np.pad(image,[[0,0],[pad,pad],[0,0]], 'constant')    
        new_center += pad
        # crop image with new centers        
        im_with = img_pad[:lowest,new_center-width//2:new_center+width//2]
        # remove background a second time because we shifted the window
        im_without = remove_background(im_with)
        plt.imshow(im_without)
        plt.show()

        # save both versions in the respective directory
        Image.fromarray(im_with).save(out_with)
        Image.fromarray(im_without).save(out_without)

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
    #root = "/net/projects/scratch/summer/valid_until_31_January_2020/asparagus/Images/unlabled/"
    root = "C:/Users/Sophia/Documents/GitHub/asparagus/Rost/"
    # get valid file names
    valid_triples = get_valid_triples(root)

    #outpath = "/net/projects/scratch/summer/valid_until_31_January_2020/asparagus/Images/preprocessed"
    # where to save the output
    # NOTE: remember to put / at the end for the subfolders
    outpath = "C:/Users/Sophia/Documents/GitHub/asparagus/Rost/preprocessed/"
    #files.sort()
    
    file_id = 0
    current_outfolder = -1
    files_per_folder = 10

    for triple in valid_triples:
        if file_id % files_per_folder == 0:
            current_outfolder +=1
        out = outpath+str(current_outfolder)+"/"
        preprocess(triple,out,file_id)
        file_id += 1
