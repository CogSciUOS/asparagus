'''these are the pca methods for the features'''

import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.spatial.distance import cdist
import os
import sys
import shutil
from grid import*
from submit_feature_pca import*

'''start with the m_hollow'''


img_shape = (1340, 364, 3)


def calculate_PC(matrix):

    #load data
    #m_hollow = np.load('/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/m_hollow.npy')
    print(matrix.shape) # das passt soweit
    #show 2 images of the data
    #plt.figure(figsize=(img_shape[0]/100,img_shape[1]/100)) #image size =  (1376, 1040)
    #plt.figure(figsize=(img_shape[0]/100,img_shape[1]/100))
    #testbild = m_hollow[1,:]
    #testbild_trans = np.reshape(testbild, newshape = (img_shape[0],img_shape[1],img_shape[2]))
    #plt.imshow(testbild_trans)# cmap = 'grey') das funzt nicht
    #plt.axis('off')
    #plt.show()

    #plt.figure(figsize=(img_shape[0]/100,img_shape[1]/100)) #image size =  (1376, 1040)
    #plt.figure(figsize=(img_shape[0]/100,img_shape[1]/100))
    #testbild2 = m_hollow[202,:]
    #testbild2_trans = np.reshape(testbild2, newshape = (img_shape[0], img_shape[1], img_shape[2]))
    #plt.imshow(testbild2_trans)# vmin=0, vmax=255 cmap = 'grey')
    #plt.axis('off')
    #plt.show()

    #standardization of the matrix
    matrix_std = (matrix - matrix.mean())/matrix.std()
    print('ist matrix_std complex? \n', np.iscomplex(matrix_std))

    #Compute eigenvectors and values
    # Covariance
    np.set_printoptions(precision=3)
    cov = np.cov(matrix_std)
    # Eigen Values
    EigVal,EigVec = np.linalg.eig(cov)
    print("Eigenvalues:\n\n", EigVal,"\n")
    print('ist Eigval complex? \n', np.iscomplex(EigVal))
    print('ist Eigvec complex? \n', np.iscomplex(EigVec))



    # Ordering Eigen values and vectors
    order = EigVal.argsort()[::-1]
    EigVal = EigVal[order]
    EigVec = EigVec[:,order]

    #calculate principle components
    PC = EigVec.T @ matrix_std
    print(PC.shape)
    print('ist PC complex? \n', np.iscomplex(PC))


    #plot the first 10 eigenvalues to get an overview
    x = range(10)
    np.linspace(0,130, 1)
    plt.plot(x,EigVal[:10])
    plt.show()
    plt.savefig('/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/data_width/eigenvalues.png')


    #  wir müssen rgb gbr umrechnung bedenken! - hint [,,::-1]
    #look at the first 10 principle components
    for i in range(10): #wir gucken uns die ersten 4 an, weil dort noch hohe eigenvalues zu sehen waren
        test = PC[i,:].reshape(img_shape)
        plt.imshow(test)
        plt.show()
        plt.savefig('/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/data_width/pca_'+str(i)+'.png')

    num_eigenvectors = 4 #lets see how many good ones we have

    eig_used = PC[:num_eigenvectors,:] #Eigenvektoren, die wir benutzen

    print("Eig_used: \n", eig_used.shape) #(4, 1463280)

    space = (matrix - matrix_std) @ eig_used.T #(400, 4)
    print("dim aspa_space: \n" , space.shape)

    #save the data
    np.save(os.path.join('/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/data_width','m_width_space'),space)
    np.save(os.path.join('/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/data_width','m_width_std'), matrix_std)
    np.save(os.path.join('/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/data_width','eig_width_used'), eig_used)
    np.save(os.path.join('/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/data_width','PC_width'), PC)


#matrix = np.load('Z:/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/data_blume/m_blume.npy')
#calculate_PC(matrix)
def calculate_first_eigval(matrix):
    #standardization of the matrix
    matrix_std = (matrix - matrix.mean())/matrix.std()
    print('ist matrix_std complex? \n', np.iscomplex(matrix_std))

    #Compute eigenvectors and values
    # Covariance
    np.set_printoptions(precision=3)
    cov = np.cov(matrix_std)
    # Eigen Values
    EigVal,EigVec = np.linalg.eig(cov)

    # Ordering Eigen values and vectors
    order = EigVal.argsort()[::-1]
    EigVal = EigVal[order]
    EigVal = EigVal[:10]

    print(EigVal)
    np.save(os.path.join('/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/data_hollow','eigval_hollow'),EigVal)



if __name__ == '__main__':
    args = typecast(sys.argv[1:])
    matrix = np.load(args[0])
    #calculate_PC(matrix)

    calculate_first_eigval(matrix)

    # args = typecast(sys.argv[1:])
    # path_to_imgs = args[0]
    # path_features = args[1]
    #
    # ids_hollow = []
    #
    # # get image_size:
    # #img = cv2.imread('Z:/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/labeled_with_background/0_b.png')
    # #print(img.shape) (1340, 364, 3)
    #
    # get_images(ids_hollow)
