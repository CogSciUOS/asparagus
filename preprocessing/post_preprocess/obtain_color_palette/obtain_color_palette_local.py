import os
import pickle
from PIL import Image
import sys
import numpy as np


def main():
    root = '/net/projects/scratch/summer/valid_until_31_January_2020/asparagus/preprocessed_images/with_background_rotated1/'
    outfile = "/net/projects/scratch/winter/valid_until_31_July_2020/mgerstenberg/asparagus/code/preprocess/post_preprocess/obtain_color_palette/colors"
    break_at_index = 1000

    colors = []
    for gridjob_folder in os.listdir(root):
        if os.path.isfile(gridjob_folder):
            continue
        for folder in os.listdir(os.path.join(root,gridjob_folder)):
            if os.path.isfile(folder):
                continue
            full_folder = os.path.join(os.path.join(root,gridjob_folder),folder)

            for i, file in enumerate(os.listdir(full_folder)):
                if i % 10 == 0:
                   print(i)
                   sys.stdout.flush()
                if i % break_at_index == 0 and i != 0:
                   with open(outfile, "wb") as f:
                        pickle.dump(colors,f)
                   sys.exit()
                idx = int(file[:-6])
                perspective = file[-5]
                filepath = os.path.join(full_folder,file)
                im = Image.open(filepath)
                im = im.convert("RGB")
                
                colors.append(im.getcolors(im.size[0]*im.size[1]))
                
                
if __name__ == "__main__":
    main()
