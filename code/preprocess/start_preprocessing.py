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

files = []
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

def preprocess(fpath1,fpath2,fpath3,outpath,file_id):    
    print(".",end="")
    sys.stdout.flush()
    
    os.makedirs(outpath, exist_ok=True)#Make dir if non existant
    
    imgs = []# open images
    imgs.append(Image.open(fpath1))
    imgs.append(Image.open(fpath2))
    imgs.append(Image.open(fpath3))
    

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
        y,x = np.array(center_of_mass(im[:,:,0]),dtype=np.int32)
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
    raw = img_array
    raw = raw[:,:,0:3]
    hsv = matplotlib.colors.rgb_to_hsv(raw)

    mask = np.logical_and(hsv[:,:,0]>0.4 , hsv[:,:,0]<0.8)#Mask all blue hues
    mask = np.logical_or(hsv[:,:,2]<80,mask)#Mask out values that are not bright engough
    
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
    root = "/net/projects/scratch/summer/valid_until_31_January_2020/asparagus/Images/unlabled/"
    files = get_files(root,".bmp","before2019" )
    print(files[:10])

    outpath = "/net/projects/scratch/summer/valid_until_31_January_2020/asparagus/Images/preprocessed"
    files.sort()
    
    n_missing = 0
    fnames = []
    
    file_id = 0
    current_outfolder = -1
    files_per_folder = 10000
    
    for file in files:
        if re.match("[0-9A-Z]+-[0-9A-Z]+-[0-9A-Z]+-[0-9A-Z]+_F00\.bmp", file):
            try:
                if file_id % files_per_folder == 0:
                    current_outfolder +=1
                root_out = outpath+"/"+str(current_outfolder)+"/"
                preprocess(root+file,root+file[:-6]+"01.bmp",root+file[:-6]+"02.bmp",root_out, file_id)
                
                file_id +=1

                    
            except FileNotFoundError:
                #print("Missing files for other perspectives for file:")
                print("-",end="")
                sys.stdout.flush()
                n_missing += 1
                fnames.append(file)   
                
            except Exception as e:
                print(file)
                print(e)
                print(traceback.format_exc())

    print("Missing " + str(n_missing))
    with open("missing.txt","wb") as f:
        pickle.dump(fnames,f)
