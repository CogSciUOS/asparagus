import os
import glob
import pandas as pd
import numpy as np

def get_asparagus_ids(PATH):
    '''
    Get ids of the asparagus pieces that have been labeled yet.
    Args: path to the combined.csv file that contains all labelfiles
    Out: only the ids aka the first column
    '''
    # read in the file
    csvs = pd.read_csv(PATH, sep = ';')
    # the column corresponding to the ids
    ids = csvs['id']
    # make it a numpy array for better parsing
    ids = np.array(ids)
    return ids

def get_files(PATH):
    '''
    Get all file names in directories and subdirectories.
    Args: PATH to files
    Out: List of all file names and the corresponding directories
    '''
    all_files = []
    file_names = []
    for subdir, dirs, files in os.walk(PATH):
        for file in files:
            filepath = subdir + '/' + file
            if filepath.endswith(".png"):
                all_files.append(filepath)
                file_names.append(file)
    return all_files, file_names


if __name__ == '__main__':
    path_to_imgs = 'Z:/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/with_background_pngs/'
    path_to_csv = 'C:/Users/Sophia/Documents/asparagus/code/variational_auto_encoder/LabelFiles/combined.csv'
    possible_folders = ['0/0','0/1','0/2','0/3','0/4','0/5','0/6', '2/0','2/1','2/2','2/3','2/4','2/5','2/6','4/0','4/1','4/2','4/3','4/4','4/5','4/6']
    # read ids from combined.csv
    ids = get_asparagus_ids(path_to_csv)
    print(len(ids))
    count = 0
    # # go through the folders that we labeled
    for folder in possible_folders:
        path = path_to_imgs + folder
        files_temp, file_names_temp = get_files(path_to_imgs)
        for name in file_names_temp:
            for i in range(len(ids)):
                if name[:-6] == str(ids[i]):
                    count += 1
                    break
        #if int(name[:-6]) in ids:
        print(count)
    # print(count)

        #all_files.append(all_files_temp)
        #file_names.append(file_names_temp)

