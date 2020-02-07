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
import cv2

ids_hollow = []

def get_asparagus_ids(PATH):
    '''
    Get ids of the asparagus pieces that have been labeled so far.
    Args: path to the combined.csv file that contains all labelfiles
    Out: only the ids aka the first column
    '''
    # read in the file
    csvs = pd.read_csv(PATH, sep = ';')

    '''get hollow and unhollow ids'''
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

# #zum testen jetzt hier rein
#     img_shape = (1340, 364, 3)
#     M_hollow = np.zeros((img_shape[0],img_shape[1]*img_shape[2],400))
#     ids_hollow = ids_hollow[:200]
#
# #store all pictures, that hollow and not hollow in a M_hollow
#     for i in ids_hollow:
#         img = cv2.imread('Z:/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/labeled_with_background/'+str(i)+'_b.png')
#         flat = np.reshape(img,newshape = (img_shape[0],img.shape[1]*img.shape[2]))
#         M_hollow[:,:,i] = flat
#         print(M_hollow)
#         return MB_img


    return ids_hollow, ids_unhollow, ids_blume, ids_notblume, ids_has_rost_head, ids_not_has_rost_head, ids_has_rost_body, ids_not_has_rost_body, ids_is_bended, ids_not_is_bended, ids_is_violet, ids_not_is_violet, ids_auto_length_big, ids_auto_length_small, ids_auto_width_big, ids_auto_width_small
#    print(len(ids_hollow))

# get image_size:
#img = cv2.imread('Z:/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/labeled_with_background/0_b.png')
#print(img.shape) (1340, 364, 3)



def get_images(ids_hollow):
    #, ids_unhollow, ids_blume, ids_notblume, ids_has_rost_head, ids_not_has_rost_head, ids_has_rost_body, ids_not_has_rost_body, ids_is_bended, ids_not_is_bended, ids_is_violet, ids_not_is_violet, ids_auto_length_big, ids_auto_length_small, ids_auto_width_big, ids_auto_width_small
     '''
     Get all images in the directories,
     referring to our feature_ids
     Args: PATH to files
     Out: matrices that store pictures of certain features
     M_hollow
     M_blume
     M_rost_head
     M_rost_body
     M_bended
     M_violet
     M_length
     M_width
     '''
     #initialize all goal matrices
     # image shape is (1340, 364, 3)
     img_shape = (1340, 364, 3)
     m_hollow = np.zeros((400, img_shape[0],img_shape[1]*img_shape[2]))
     M_hollow = np.zeros((200, img_shape[0],img_shape[1]*img_shape[2]))
     M_unhollow = np.zeros((200, img_shape[0],img_shape[1]*img_shape[2]))
#     M_blume = np.zeros((img_shape[0],img_shape[1]*img_shape[2],400)) # 3448 ist leider zu groß...
#     M_rost_head = np.zeros((img_shape[0],img_shape[1]*img_shape[2],400))#3910
#     M_rost_body = np.zeros((img_shape[0],img_shape[1]*img_shape[2],400))#12158
#     M_bended = np.zeros((img_shape[0],img_shape[1]*img_shape[2],400))#10674
#     M_violet = np.zeros((img_shape[0],img_shape[1]*img_shape[2],400))
#     M_length = np.zeros((img_shape[0],img_shape[1]*img_shape[2],200))
#     M_width = np.zeros((img_shape[0],img_shape[1]*img_shape[2],200))#8798
     #store all pictures, that hollow and not hollow in a M_hollow
     ids_hollow, ids_unhollow, ids_blume, ids_notblume, ids_has_rost_head, ids_not_has_rost_head, ids_has_rost_body, ids_not_has_rost_body, ids_is_bended, ids_not_is_bended, ids_is_violet, ids_not_is_violet, ids_auto_length_big, ids_auto_length_small, ids_auto_width_big, ids_auto_width_small = get_asparagus_ids("combined_new.csv")
     print(ids_hollow)
     ids_hollow = ids_hollow[:200]
     ids_unhollow = ids_unhollow[:200]
     print(ids_hollow)
     print(ids_unhollow)


     #store all pictures, that hollow and not hollow in a M_hollow
     for i in ids_hollow:
         print(ids_hollow[i])
         img = cv2.imread('Z:/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/labeled_with_background/'+str(i)+'_b.png')
         flat = np.reshape(img,newshape = (img_shape[0],img.shape[1]*img.shape[2]))
         M_hollow[i,:,:] = flat #Error index 207 is out of bounds for axis 0 with size 200: er geht hier komischerweise bis bild mit dem namen 190 und dann würde bild 207 kommen...
         print(M_hollow)
#         return M_hollow

     for i in ids_unhollow:
         img = img = cv2.imread('Z:/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/labeled_with_background/'+str(i)+'_b.png')
         flat = np.reshape(img,newshape = (img_shape[0],img.shape[1]*img.shape[2]))
         M_unhollow[i,:,:] = flat
#         return M_unhollow
#put hollow and unhollow together:
     m_hollow = np.concatenate(M_hollow,M_unhollow)
     print(M_hollow.shape)


    #     for c in M_hollow[2]:
#             if cv2.imread('Z:/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/labeled_with_background/'+str(i)+'_b.png' == ids_hollow[1]:
#                 img = cv2.imread('Z:/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/labeled_with_background/'+str(i)+'_b.png'
#                 flat = np.reshape(img,newshape = (img_shape[0],img.shape[1]*img.shape[2]))
#                 MB_img[:,:,i] = flat
#                 return MB_img
get_images(ids_hollow)
#       return M_hollow, M_blume, M_rost_head, M_rost_body, M_bended, M_violet, M_length
#     all_files = []
#     file_names = []
#     for subdir, dirs, files in os.walk(PATH):
#         for file in files:
#             filepath = subdir + '/' + file
#             if filepath.endswith(".png"):
#                 all_files.append(filepath)
#                 file_names.append(int(file[:-6])) #modified to not save without _a.png and save as int for later comparison
#     return all_files, file_names
#get_images(ids_hollow)
#get_images(PATH, ids_hollow, ids_unhollow, ids_blume, ids_notblume, ids_has_rost_head, ids_not_has_rost_head, ids_has_rost_body, ids_not_has_rost_body, ids_is_bended, ids_not_is_bended, ids_is_violet, ids_not_is_violet, ids_auto_length_big, ids_auto_length_small, ids_auto_width_big, ids_auto_width_small)
