'''This is a PCA considering all features, that we have
Features are:
is_bruch
is_hollow
has_blume
has_rost_head
has_rost_body
is_bended
is_violet
auto_length : these are values that were not labeled manually
auto_width  : these are values that were not labeled manually
'''

'''For each feature, the image undergoes the pca method
as output we get for the feature a binary score.
Eg.: is_bruch = TRUE or is_bruch = FALSE

all the calculated values need to be put in the method that gives
the asparagus a class'''


#get how many hollow and not hollow? 100/100, oder 500/500?
# auf welchen index geht die csv original files to index?

#hier muss so ein loop: if value  in bestimmter spate == 1, dann extracte erstmal den path, aber nur ganz hinten....
# 0/0/999_b - ist das letzte aus dem 0er ordner
# aber es passt soweit alles. also dann nur die nummer extracten, dann diese nummer in Z:\net\projects\scratch\winter\valid_until_31_July_2020\asparagus\preprocessed_images\labeled_with_background
# raussuchen (da können wir eigentlich auch einfach a,b,c nehmen)

import pandas as pd
import numpy as np
import os
#from grid import*
import sys
import shutil
import itertools

def get_asparagus_ids(PATH):
    '''
    Get ids of the asparagus pieces that have been labeled so far.
    Args: path to the combined.csv file that contains all labelfiles
    Out: only the ids aka the first column
    '''
    # read in the file
    csvs = pd.read_csv(PATH, sep = ';')

    '''get hollow and unhollow ids'''
    ids_hollow = []
    ids_hollow = csvs.loc[csvs['is_hollow']== 1.0, 'id']
    print(ids_hollow) # die nummerierung ist ids[1] also die richtige column
    # wir haben 439 hollow

    ids_unhollow = []
    ids_unhollow = csvs.loc[csvs['is_hollow']== 0.0, 'id']
    ids_unhollow = ids_unhollow[:439]
    print(ids_unhollow)

    '''get is_bruch ids

    ids_bruch = []
    ids_hollow = csvs.loc[csvs['is_bruch']== 1.0, 'id']
    print(ids_hollow) # die nummerierung ist ids[1] also die richtige column

    these are only 5 asparagus, so we wont use this feature'''


    '''get has_blume and not_blume'''
    ids_blume = []
    ids_blume = csvs.loc[csvs['has_blume']== 1.0, 'id']
    print(ids_blume)

    ids_notblume = []
    ids_notblume = csvs.loc[csvs['has_blume']== 0.0, 'id']
    ids_notblume = ids_notblume[:1724]
    print(ids_notblume)

    '''get has_rost_head and not_has_rost_head'''
    ids_has_rost_head = []
    ids_has_rost_head = csvs.loc[csvs['has_rost_head']== 1.0, 'id']
    print(ids_has_rost_head)

    ids_not_has_rost_head = []
    ids_not_has_rost_head = csvs.loc[csvs['has_blume']== 0.0, 'id']
    ids_not_has_rost_head = ids_not_has_rost_head[:1955]
    print(ids_not_has_rost_head)

    '''get has_rost_body and not_has_rost_body'''
    ids_has_rost_body = []
    ids_has_rost_body = csvs.loc[csvs['has_rost_body']== 1.0, 'id']
    print(ids_has_rost_body)

    ids_not_has_rost_body = []
    ids_not_has_rost_body = csvs.loc[csvs['has_rost_body']== 0.0, 'id']
    ids_not_has_rost_body = ids_not_has_rost_body[:6079]
    print(ids_not_has_rost_body)

    '''get is_bended and not_is_bended'''
    ids_is_bended = []
    ids_is_bended = csvs.loc[csvs['is_bended']== 1.0, 'id']
    print(ids_is_bended)

    ids_not_is_bended = []
    ids_not_is_bended = csvs.loc[csvs['is_bended']== 0.0, 'id']
    ids_not_is_bended = ids_not_is_bended[:5337]
    print(ids_not_is_bended)

    '''get is_violet and not_is_violet'''
    ids_is_violet = []
    ids_is_violet = csvs.loc[csvs['is_violet']== 1.0, 'id']
    print(ids_is_violet)

    ids_not_is_violet = []
    ids_not_is_violet = csvs.loc[csvs['is_violet']== 0.0, 'id']
    ids_not_is_violet = ids_not_is_violet[:1047]
    print(ids_not_is_violet)

    '''auto_length > 210 mm and < 210 mm'''
    ids_auto_length_big = []
    ids_auto_length_big = csvs.loc[csvs['auto_length']> 210, 'id']
    ids_auto_length_big = ids_auto_length_big[:455]
    print(ids_auto_length_big)

    ids_auto_length_small = []
    ids_auto_length_small = csvs.loc[csvs['auto_length'] <= 210, 'id']
    print(ids_auto_length_small)

    '''auto_width > 20 mm and < 20 mm'''
    ids_auto_width_big = []
    ids_auto_width_big = csvs.loc[csvs['auto_width']> 20, 'id']
#    ids_auto_width_big = ids_auto_width_big[:455]
    print(ids_auto_width_big)

    ids_auto_width_small = []
    ids_auto_width_small = csvs.loc[csvs['auto_width'] <= 20, 'id']
    ids_auto_width_small = ids_auto_width_small[:4399]
    print(ids_auto_width_small)

    return ids_hollow, ids_unhollow, ids_blume, ids_notblume, ids_has_rost_head, ids_not_has_rost_head, ids_has_rost_body, ids_not_has_rost_body, ids_is_bended, ids_not_is_bended, ids_is_violet, ids_not_is_violet, ids_auto_length_big, ids_auto_length_small, ids_auto_width_big, ids_auto_width_small
#    print(len(ids_hollow))

get_asparagus_ids("combined_new.csv")


# def get_files(PATH):
#     '''
#     Get all file names in directories and subdirectories.
#     Args: PATH to files
#     Out: List of all file names and the corresponding directories
#     '''
#     all_files = []
#     file_names = []
#     for subdir, dirs, files in os.walk(PATH):
#         for file in files:
#             filepath = subdir + '/' + file
#             if filepath.endswith(".png"):
#                 all_files.append(filepath)
#                 file_names.append(int(file[:-6])) #modified to not save without _a.png and save as int for later comparison
#     return all_files, file_names
