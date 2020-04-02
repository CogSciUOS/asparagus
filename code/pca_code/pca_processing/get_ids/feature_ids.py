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


#get how many hollow and not hollow? we choose 200/valid_until_31_July_2020

import pandas as pd
import numpy as np
import os
from grid import*
from submit_feature_ids import*
import sys
import shutil
import itertools
import cv2

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
    ids_hollow = ids_hollow[:200]
    ids_hollow = np.array(ids_hollow)
    print(ids_hollow)
 # die nummerierung ist ids[1] also die richtige column
    # wir haben 439 hollow

    ids_unhollow = []
    ids_unhollow = csvs.loc[csvs['is_hollow']== 0.0, 'id']
    ids_unhollow = ids_unhollow[:200]
    ids_unhollow = np.array(ids_unhollow)
    print(ids_unhollow)

    '''get is_bruch ids

    ids_bruch = []
    ids_hollow = csvs.loc[csvs['is_bruch']== 1.0, 'id']
    print(ids_hollow) # die nummerierung ist ids[1] also die richtige column

    these are only 5 asparagus, so we wont use this feature'''


    '''get has_blume and not_blume'''
    ids_blume = []
    ids_blume = csvs.loc[csvs['has_blume']== 1.0, 'id']
    ids_blume = ids_blume[:200]
    ids_blume = np.array(ids_blume)

    ids_notblume = []
    ids_notblume = csvs.loc[csvs['has_blume']== 0.0, 'id']
    ids_notblume = ids_notblume[:200]
    ids_notblume = np.array(ids_notblume)

    '''get has_rost_head and not_has_rost_head'''
    ids_has_rost_head = []
    ids_has_rost_head = csvs.loc[csvs['has_rost_head']== 1.0, 'id']
    ids_has_rost_head = ids_has_rost_head[:200]
    ids_has_rost_head = np.array(ids_has_rost_head)

    ids_not_has_rost_head = []
    ids_not_has_rost_head = csvs.loc[csvs['has_blume']== 0.0, 'id']
    ids_not_has_rost_head = ids_not_has_rost_head[:200]
    ids_not_has_rost_head = np.array(ids_not_has_rost_head)

    '''get has_rost_body and not_has_rost_body'''
    ids_has_rost_body = []
    ids_has_rost_body = csvs.loc[csvs['has_rost_body']== 1.0, 'id']
    ids_has_rost_body = ids_has_rost_body[:200]
    ids_has_rost_body = np.array(ids_has_rost_body)

    ids_not_has_rost_body = []
    ids_not_has_rost_body = csvs.loc[csvs['has_rost_body']== 0.0, 'id']
    ids_not_has_rost_body = ids_not_has_rost_body[:200]
    ids_not_has_rost_body = np.array(ids_not_has_rost_body)

    '''get is_bended and not_is_bended'''
    ids_is_bended = []
    ids_is_bended = csvs.loc[csvs['is_bended']== 1.0, 'id']
    ids_is_bended = ids_is_bended[:200]
    ids_is_bended = np.array(ids_is_bended)

    ids_not_is_bended = []
    ids_not_is_bended = csvs.loc[csvs['is_bended']== 0.0, 'id']
    ids_not_is_bended = ids_not_is_bended[:200]
    ids_not_is_bended = np.array(ids_not_is_bended)

    '''get is_violet and not_is_violet'''
    ids_is_violet = []
    ids_is_violet = csvs.loc[csvs['is_violet']== 1.0, 'id']
    ids_is_violet = ids_is_violet[:200]
    ids_is_violet = np.array(ids_is_violet)

    ids_not_is_violet = []
    ids_not_is_violet = csvs.loc[csvs['is_violet']== 0.0, 'id']
    ids_not_is_violet = ids_not_is_violet[:200]
    ids_not_is_violet = np.array(ids_not_is_violet)

    '''auto_length > 210 mm and < 210 mm'''
    ids_auto_length_big = []
    ids_auto_length_big = csvs.loc[csvs['auto_length']> 210, 'id']
    ids_auto_length_big = ids_auto_length_big[:200]
    ids_auto_length_big = np.array(ids_auto_length_big)

    ids_auto_length_small = []
    ids_auto_length_small = csvs.loc[csvs['auto_length'] <= 210, 'id']
    ids_auto_length_small = ids_auto_length_small[:200]
    ids_auto_length_small = np.array(ids_auto_length_small)

    '''auto_width > 20 mm and < 20 mm'''
    ids_auto_width_big = []
    ids_auto_width_big = csvs.loc[csvs['auto_width']> 20, 'id']
    ids_auto_width_big = ids_auto_width_big[:200]
    ids_auto_width_big = np.array(ids_auto_width_big)

    ids_auto_width_small = []
    ids_auto_width_small = csvs.loc[csvs['auto_width'] <= 20, 'id']
    ids_auto_width_small = ids_auto_width_small[:200]
    ids_auto_width_small = np.array(ids_auto_width_small)


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

def get_images(ids_hollow):

    # nacheinander ids_unhollow, ids_blume, ids_notblume, ids_has_rost_head, ids_not_has_rost_head, ids_has_rost_body, ids_not_has_rost_body, ids_is_bended, ids_not_is_bended, ids_is_violet, ids_not_is_violet, ids_auto_length_big, ids_auto_length_small, ids_auto_width_big, ids_auto_width_small
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
     #get all ids
     ids_hollow, ids_unhollow, ids_blume, ids_notblume, ids_has_rost_head, ids_not_has_rost_head, ids_has_rost_body, ids_not_has_rost_body, ids_is_bended, ids_not_is_bended, ids_is_violet, ids_not_is_violet, ids_auto_length_big, ids_auto_length_small, ids_auto_width_big, ids_auto_width_small = get_asparagus_ids(path_features)
    #initialize all goal matrices
     # image shape is (1340, 364, 3)
     img_shape = (1340, 364, 3)


     #initialize
     all_ids_hollow = np.concatenate((ids_hollow, ids_unhollow))
     m_hollow = np.zeros((400, img_shape[0]*img_shape[1]*img_shape[2]))
     #M_hollow = np.zeros((n_bands, img_shape[0]*img_shape[1]*img_shape[2]))
     #M_unhollow = np.zeros((n_bands, img_shape[0]*img_shape[1]*img_shape[2]))
     #print(all_ids_hollow)

     #all_ids_blume = np.concatenate((ids_blume, ids_notblume))
     #m_blume = np.zeros((400, img_shape[0]*img_shape[1]*img_shape[2]))
     #M_blume = np.zeros((n_bands, img_shape[0]*img_shape[1]*img_shape[2]))
     #M_not_blume = np.zeros((n_bands, img_shape[0]*img_shape[1]*img_shape[2]))

     # all_ids_rost_head = np.concatenate((ids_has_rost_head,ids_not_has_rost_head))
     # m_rost_head = np.zeros((400,img_shape[0]*img_shape[1]*img_shape[2]))
#     M_rost_head = np.zeros((n_bands, img_shape[0]*img_shape[1]*img_shape[2]))
#     M_not_rost_head = np.zeros((n_bands, img_shape[0]*img_shape[1]*img_shape[2]))

     #all_ids_rost_body = np.concatenate((ids_has_rost_body, ids_not_has_rost_body))
     #m_rost_body = np.zeros((400, img_shape[0]*img_shape[1]*img_shape[2]))
#     M_rost_body = np.zeros((n_bands, img_shape[0]*img_shape[1]*img_shape[2]))
#     M_not_rost_body = np.zeros((n_bands, img_shape[0]*img_shape[1]*img_shape[2]))

     #all_ids_bended = np.concatenate((ids_is_bended, ids_not_is_bended))
     #m_bended = np.zeros((400, img_shape[0]*img_shape[1]*img_shape[2]))
     #M_bended = np.zeros((n_bands, img_shape[0]*img_shape[1]*img_shape[2]))#10674
     #M_not_bended = np.zeros((n_bands, img_shape[0]*img_shape[1]*img_shape[2]))

     #all_ids_violet = np.concatenate((ids_is_violet, ids_not_is_violet))
     #m_violet = np.zeros((400, img_shape[0]*img_shape[1]*img_shape[2]))
#     M_violet = np.zeros((n_bands,img_shape[0]*img_shape[1]*img_shape[2]))
#     M_not_violet = np.zeros((n_bands,img_shape[0]*img_shape[1]*img_shape[2]))

     #all_ids_length = np.concatenate((ids_auto_length_big, ids_auto_length_small))
     #m_length = np.zeros((400,img_shape[0]*img_shape[1]*img_shape[2])) #hier geht schon der speicher aus
#     M_length_big = np.zeros((n_bands, img_shape[0]*img_shape[1]*img_shape[2]))
#     M_length_small = np.zeros((n_bands, img_shape[0]*img_shape[1]*img_shape[2]))

     #all_ids_width = np.concatenate((ids_auto_width_big, ids_auto_width_small))
     #m_width = np.zeros((400,img_shape[0]*img_shape[1]*img_shape[2]))
#     M_width_big = np.zeros((n_bands, img_shape[0]*img_shape[1]*img_shape[2]))
#     M_width_small = np.zeros((n_bands, img_shape[0]*img_shape[1]*img_shape[2]))#8798
     #store all pictures, that hollow and not hollow in a M_hollow

     #fill matries with pictures

    # fill m_hollow
     s = 0
     for i in all_ids_hollow:
       img = cv2.imread(path_to_imgs+str(i)+'_b.png')
       #print(img.shape)
       flat = np.reshape(img,newshape = (img_shape[0]*img_shape[1]*img_shape[2]))
       m_hollow[s,:] = flat
       s += 1

     #np.save('m_hollow',m_hollow)
     np.save(os.path.join('/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/data_hollow','m_hollow'), m_hollow)

     #fill M_blume
     #s = 0
     #for i in all_ids_blume:
    #     img = cv2.imread('/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/labeled_with_background/'+str(i)+'_b.png')
    #     #print(img.shape)
    #     flat = np.reshape(img,newshape = (img_shape[0]*img_shape[1]*img_shape[2]))
    #     m_blume[s,:] = flat
    #     s += 1
    #
    # np.save(os.path.join('/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images','m_blume'),m_blume)
    #
    #  #fill m_rost_head
    #  s = 0
    #  for i in all_ids_rost_head:
    #      img = cv2.imread('/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/labeled_with_background/'+str(i)+'_b.png')
    #      #print(img.shape)
    #      flat = np.reshape(img,newshape = (img_shape[0]*img_shape[1]*img_shape[2]))
    #      m_rost_head[s,:] = flat
    #      s += 1
    # #
    #  np.save(os.path.join('/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images','m_rost_head'), m_rost_head)
    # #
    #  #fill m_rost_body
     #s = 0
     #for i in all_ids_rost_body:
    #     img = cv2.imread('/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/labeled_with_background/'+str(i)+'_b.png')
    ##     #print(img.shape)
    #     flat = np.reshape(img,newshape = (img_shape[0]*img_shape[1]*img_shape[2]))
    #     m_rost_body[s,:] = flat
    #     s += 1
    #
     #np.save(os.path.join('/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images','m_rost_body'),m_rost_body)
    #
    #  #fill all_ids M_bended
    # s = 0
     #for i in all_ids_bended:
    #      img = cv2.imread('/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/labeled_with_background/'+str(i)+'_b.png')
    #      flat = np.reshape(img,newshape = (img_shape[0]*img_shape[1]*img_shape[2]))
    #      m_bended[s,:] = flat
    #      s += 1
    #
     #np.save(os.path.join('/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images','m_bended'),m_bended)
    #
    # #fill all_ids_violet
     #s = 0
     #for i in all_ids_violet:
    #     img = cv2.imread('/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/labeled_with_background/'+str(i)+'_b.png')
    #     flat = np.reshape(img,newshape = (img_shape[0]*img_shape[1]*img_shape[2]))
    #     m_violet[s,:] = flat
    #     s += 1
    #
     #np.save(os.path.join('/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images','m_violet'),m_violet)
    #fill auto_length
     #s = 0
     #for i in all_ids_length:
    #     img = cv2.imread('/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/labeled_with_background/'+str(i)+'_b.png')
    #     flat = np.reshape(img,newshape = (img_shape[0]*img_shape[1]*img_shape[2]))
    #     m_length[s,:] = flat
    #     s += 1
    #
    # np.save(os.path.join('/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images','m_length'),m_length)
    #
    # #fill width
     #s = 0
     #for i in all_ids_width:
    #     img = cv2.imread('/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/labeled_with_background/'+str(i)+'_b.png')
    #     flat = np.reshape(img,newshape = (img_shape[0]*img_shape[1]*img_shape[2]))
    #     m_width[s,:] = flat
    #     s += 1
    ##
     #np.save(os.path.join('/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images','m_width'),m_width)
    #

     #return m_rost_head


if __name__ == '__main__':
    args = typecast(sys.argv[1:])
    path_to_imgs = args[0]
    path_features = args[1]

    ids_width = []

    # get image_size:
    #img = cv2.imread('Z:/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/labeled_with_background/0_b.png')
    #print(img.shape) (1340, 364, 3)

    get_images(ids_width)
