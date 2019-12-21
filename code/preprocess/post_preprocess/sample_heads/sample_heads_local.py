import os
import pickle
from PIL import Image
import sys
import numpy as np

def sample_head(img, width=150,height=150, output_height=64,output_width=64):
    img = np.array(img)
    raw_img = img.copy()
    img = img[:,:,0]
    asparagus = img>0
    y_mean = np.any(img,axis=1)
    y = np.argmax(y_mean>0)
    x = np.mean(np.where(img[y]>0))
    left = int(max(x-width//2,0))
    top = int(min(y,img.shape[0]-width))
    head = Image.fromarray(raw_img[ top  : top+height,
                                left : left+width]).resize((output_height,output_width))
    return head

def main():
    root = '/net/projects/scratch/summer/valid_until_31_January_2020/asparagus/preprocessed_images/with_background_rotated1/'
    outroot = '/net/projects/scratch/summer/valid_until_31_January_2020/asparagus/preprocessed_images/without_background_heads/'

    for gridjob_folder in os.listdir(root):
        if os.path.isfile(gridjob_folder):
            continue
        for folder in os.listdir(os.path.join(root,gridjob_folder)):
            if os.path.isfile(folder):
                continue
            full_folder = os.path.join(os.path.join(root,gridjob_folder),folder)
            full_outfolder = os.path.join(os.path.join(outroot,gridjob_folder),folder)
            os.makedirs(full_outfolder, exist_ok=True)

            for i, file in enumerate(os.listdir(full_folder)):
                if i % 100 == 0:
                   print(i)
                   sys.stdout.flush()
                idx = int(file[:-6])
                perspective = file[-5]
                filepath = os.path.join(full_folder,file)
                outpath = os.path.join(full_outfolder,file)

                im = Image.open(filepath)
                im = sample_head(im)
                im.save(outpath)
if __name__ == "__main__":
    main()
