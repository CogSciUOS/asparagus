'''these are the pca methods for the features'''

import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.spatial.distance import cdist


'''start with the m_hollow'''


img_shape = (1340, 364, 3)

def calculate_PC(m_hollow):

    #load data
    m_hollow = np.load('Z:/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/m_hollow.npy')
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
    m_hollow_std = (m_hollow - m_hollow.mean())/m_hollow.std()


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
    PC = EigVec.T @ m_hollow_std
    print(PC.shape)

    #plot the first 10 eigenvalues to get an overview
    x = range(10)
    np.linspace(0,130, 1)
    plt.plot(x,EigVal[:10])
    plt.show()

    #look at the first 10 principle components
    #for i in range(10): #wir gucken uns die ersten 4 an, weil dort noch hohe eigenvalues zu sehen waren
    #    test = PC[i,:].reshape(img_shape)
    #    plt.imshow(test)
    #    plt.show()

    num_eigenvectors = 4 #lets see how many good ones we have

    eig_hollow_used = PC[:,:num_eigenvectors] #Eigenvektoren, die wir benutzen
    print("Eig_hollow_used: \n", eig_hollow_used.shape)

    hollow_space = (m_hollow - m_hollow_std) @ eig_hollow_used.T
    print("dim aspa_space: \n" , hollow_space.shape)

    np.save('hollow_space',hollow_space)
    np.save('m_hollow_std', m_hollow_std)
    np.save('eig_hollow_used', eig_hollow_used)

    return hollow_space, m_hollow_std, eig_hollow_used

bla = []
calculate_PC(bla)


    # def recognize_face(face, eigenfaces, mean_face, face_db):
    #     """
    #     Recognize a face from a face database.
    #     and return the index of the best matching database entry.
    #
    #     The FACE is first centered and projected into the eigeface
    #     space provided by EIGENFACES. Then the best match is found
    #     according to the euclidean distance in the eigenface space.
    #
    #     Args:
    #         face (ndarray): Face to be recognised.
    #         eigenfaces (ndarray): Array of eigenfaces.
    #         mean_face (ndarray): Average face.
    #         face_db (ndarray): Database of faces projectected into Eigenface space.
    #
    #     Returns:
    #         index (uint): Position of the best matching face in face_db.
    #     """
    #     index = -1
    #
    #     # BEGIN SOLUTION
    #     # center the face
    #     #face = np.reshape(img,newshape = (img_shape[0],img.shape[1]*img.shape[2]))
    #     #face_img = face #np.zeros((img_shape[0],img.shape[1]*img.shape[2]))  #(1376, 1040)
    #     #face_img = face_img.flatten()
    #     #print(face_img.shape)
    #
    #
    #     #centered = face_img - MB_matrix_mean
    #     centered = face - MB_matrix_mean
    #
    #     #centered = centered.flatten()
    #     print(centered.shape)#(1340, 1092),durch flatten (1463280,)
    #     print(eigenfaces.shape) #(1463280, 130)
    #
    #     # and project it into the eigenface space
    #     projected = np.matmul(centered, eigenfaces)
    #     #print(projected.shape) #(130,)
    #
    #     # Now compute the similarity to all known faces
    #     # (comparison is performed in the eigenface space)
    #     #projected = projected.T #diese beiden operationen brauchte ich, wenn ich
    #     #face_db = face_db.T    #die funktion auf nur ein bild angewand habe, jetzt mit der großen funktion scheint es überflüssig
    #     #print('jetzt mit großem input face_db: \n',face_db.shape) #(1463280, 4)(130, 4)
    #     #print('jetzt mit großem input projected: \n',projected.shape) #(1463280, 4)(130, 4)
    #
    #     distances = cdist(face_db, projected[None, :])#[None, :] das war direkt an projected
    #     index = np.argmin(distances)
    #     #return index
    # #    index = int(round(index /10))
    #     #intex = int(index)
    #     print('index: \n',index)
    #     # END SOLUTION
    #
    #     return index
