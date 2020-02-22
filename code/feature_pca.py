'''these are the pca methods for the features'''

import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.spatial.distance import cdist
import os

'''start with the m_hollow'''


img_shape = (1340, 364, 3)

def calculate_PC(m_hollow):

    #load data
    m_hollow = np.load('/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/m_hollow.npy')
    print(m_hollow.shape) # das passt soweit
    #show 2 images of the data
    #plt.figure(figsize=(img_shape[0]/100,img_shape[1]/100)) #image size =  (1376, 1040)
    plt.figure(figsize=(img_shape[0]/100,img_shape[1]/100))
    testbild = m_hollow[1,:]
    testbild_trans = np.reshape(testbild, newshape = (img_shape[0],img_shape[1],img_shape[2]))
    plt.imshow(testbild_trans)# cmap = 'grey') das funzt nicht
    plt.axis('off')
    plt.show()

    #plt.figure(figsize=(img_shape[0]/100,img_shape[1]/100)) #image size =  (1376, 1040)
    plt.figure(figsize=(img_shape[0]/100,img_shape[1]/100))
    testbild2 = m_hollow[202,:]
    testbild2_trans = np.reshape(testbild2, newshape = (img_shape[0], img_shape[1], img_shape[2]))
    plt.imshow(testbild2_trans)# vmin=0, vmax=255 cmap = 'grey')
    plt.axis('off')
    plt.show()

    #standardization of the matrix
    m_hollow_std = (m_hollow - m_hollow.mean(axis = 0))/m_hollow.std()


    #Compute eigenvectors and values
    # Covariance
    np.set_printoptions(precision=3)
    cov = np.cov(m_hollow_std)
    # Eigen Values
    EigVal,EigVec = np.linalg.eig(cov)
    print("Eigenvalues:\n\n", EigVal,"\n")

    # Ordering Eigen values and vectors
    order = EigVal.argsort()[::-1]
    EigVal = EigVal[order]
    EigVec = EigVec[:,order]

    #calculate principle components
    PC_hollow = EigVec.T @ m_hollow_std
    print(PC_hollow.shape)

    #plot the first 10 eigenvalues to get an overview
    x = range(10)
    np.linspace(0,130, 1)
    plt.plot(x,EigVal[:10])
    plt.show()


    #  wir müssen rgb gbr umrechnung bedenken! - hint [,,::-1]
    #look at the first 10 principle components
    #for i in range(10): #wir gucken uns die ersten 4 an, weil dort noch hohe eigenvalues zu sehen waren
    #    test = PC[i,:].reshape(img_shape)
    #    plt.imshow(test)
    #    plt.show()

    num_eigenvectors = 4 #lets see how many good ones we have

    #das meinte sophia, wäre dann quasi doe überbleibenden PCs, aber man kann das auch anders berechenen, vielleicht liegt es daran
    eig_hollow_used = PC_hollow[:num_eigenvectors,:] #Eigenvektoren, die wir benutzen

    print("Eig_hollow_used: \n", eig_hollow_used.shape) #(4, 1463280)

    hollow_space = (m_hollow - m_hollow_std) @ eig_hollow_used.T #(400, 4)
    print("dim aspa_space: \n" , hollow_space.shape)

    #save the data
    np.save(os.path.join('/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images','m_hollow_space_1'),hollow_space)
    np.save(os.path.join('/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images','m_hollow_std_1'), m_hollow_std)
    np.save(os.path.join('/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images','eig_hollow_used_1'), eig_hollow_used)
    np.save(os.path.join('/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images','PC_hollow_1'), PC_hollow)



if __name__ == '__main__':
    bla = []
    calculate_PC(bla)
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
