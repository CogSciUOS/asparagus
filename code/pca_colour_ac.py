'''
this is a new method for greyscale rgb, combinig the first method with the last one,
that did not work for rgb images
'''

from IPython.display import Image, display
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import cv2
import numpy as np
import matplotlib.gridspec as grid
import wx #package wx
from PIL import Image
import matplotlib.image as mpimg
from scipy.spatial.distance import cdist


#get the files (files contai 10 images of each class)

img = cv2.imread('C:/Users/schmuri/github/asparagus/code/pca_images_all_classes/0_b.png')

img_shape = img.shape #(1376, 1040, 3)

n_bands = 130 #weil ich 130 bilder habe

MB_img = np.zeros((img_shape[0],img_shape[1]*img_shape[2],n_bands))  #(1376, 1040, 7)
s = 0
# stacking up images into the array

for i in range(n_bands):
    #MB_img[:,:,i] = cv2.imread('band'+str(i+1)+'.jpg', cv2.IMREAD_GRAYSCALE)
    #das hier bedeutet, dass er jeweils drei bilder hat, in unterschiedlichen farbdingern
    #reading in all 3 images of one asparagus (unprocessed)
    img = cv2.imread('C:/Users/schmuri/github/asparagus/code/pca_images_all_classes/'+str(s+i)+'_b.png')
#wie komme ich jetzt auf diesen path...'C:/Users/schmuri/github/asparagus/code/pca_images_all_classes/*.png'
    flat = np.reshape(img,newshape = (img_shape[0],img.shape[1]*img.shape[2]))

    MB_img[:,:,i] = flat
    #cv2.imread('C:/Users/schmuri/github/asparagus/code/pca_images_all_classes/'+str(s+i)+'_b.png', cv2.IMREAD_GRAYSCALE)

#mal checken
#plt.figure(figsize=(img_shape[0]/100,img_shape[1]/100)) #image size =  (1376, 1040)
#plt.imshow(MB_img[:,:,1], vmin=0, vmax=255, cmap = 'grey')
#plt.axis('off')
#plt.show()

# #####this is Standardization
# # Convert 2d band array in 1-d to make them as feature vectors and Standardization
MB_matrix = np.zeros((MB_img[:,:,0].size,n_bands))
for i in range(n_bands):
    MB_array = MB_img[:,:,i].flatten()  # covert 2d to 1d array
    MB_arrayStd = (MB_array - MB_array.mean())/MB_array.std()
    MB_matrix[:,i] = MB_arrayStd
print(MB_matrix.shape)

print(MB_matrix)

#Compute eigenvectors and values
# Covariance
np.set_printoptions(precision=3)
cov = np.cov(MB_matrix.T)
# Eigen Values
EigVal,EigVec = np.linalg.eig(cov)
print("Eigenvalues:\n\n", EigVal,"\n")
  # Ordering Eigen values and vectors
order = EigVal.argsort()[::-1]
EigVal = EigVal[order]
EigVec = EigVec[:,order]

PC = np.matmul(MB_matrix,EigVec) # calculating principle components
num_eigenvectors = 4

eig_asparagus_used = PC[:,:num_eigenvectors] #Eigenvektoren, die wir benutzen
print("Eig_aspa_used: \n", eig_asparagus_used.shape)
#Eigen_used = EigVec[:num_eigenvectors]
#print(Eigen_used.shape) #(130,4)
#Eig_mean = EigVec.mean(axis=0)

#asparagus_db = (EigVec - Eig_mean) @ Eigen_used.T
#print(asparagus_db.shape)

#Projecting data on Eigen vector directions resulting to Principal Components


MB_matrix_mean = np.mean(MB_matrix, axis = 1)
print(MB_matrix_mean.shape)
print(MB_matrix.shape)
asparagus_space = (MB_matrix.T - MB_matrix_mean) @ eig_asparagus_used
print("dim aspa_space: \n" , asparagus_space.shape)


print(PC.shape)
#teil eins vom plotten der PCs
# for i in range(10): #wir gucken uns die ersten 4 an, weil dort noch hohe eigenvalues zu sehen waren
#      test = PC[:,i].reshape(img_shape)
#      plt.imshow(test)
#      plt.show()



#print(test.shape)#(1340, 364, 3)
#teil 2 vom plotten der PCs
# x = range(10)
# # np.linspace(0,130, 1)
# plt.plot(x,EigVal[:10])
# plt.show()

#save eigenvalues
#np.save('EigVal.npy', EigVal)
# human readable
#np.savetxt('evecs_mata.txt', evecs_mat)


from scipy.spatial.distance import cdist
#
def recognize_face(face, eigenfaces, mean_face, face_db):
    """
    Recognize a face from a face database.
    and return the index of the best matching database entry.

    The FACE is first centered and projected into the eigeface
    space provided by EIGENFACES. Then the best match is found
    according to the euclidean distance in the eigenface space.

    Args:
        face (ndarray): Face to be recognised.
        eigenfaces (ndarray): Array of eigenfaces.
        mean_face (ndarray): Average face.
        face_db (ndarray): Database of faces projectected into Eigenface space.

    Returns:
        index (uint): Position of the best matching face in face_db.
    """
    index = -1

    # BEGIN SOLUTION
    # center the face
    face = np.reshape(img,newshape = (img_shape[0],img.shape[1]*img.shape[2]))
    face_img = np.zeros((img_shape[0],img.shape[1]*img.shape[2]))  #(1376, 1040)
    face_img = face_img.flatten()
    print(face_img.shape)

    #face_array = face_img.flatten()
#    print(face_array.shape)

    centered = face_img - MB_matrix_mean
    #centered = centered.flatten()
    print(centered.shape)#(1340, 1092),durch flatten (1463280,)
    print(eigenfaces.shape) #(1463280, 130)

    # and project it into the eigenface space
    projected = np.matmul(centered, eigenfaces)
    print(projected.shape) #(130,)

    # Now compute the similarity to all known faces
    # (comparison is performed in the eigenface space)
    print(face_db.shape) #(1463280, 4)(130, 4)
    distances = cdist(face_db, projected[None, :])#[None, :] das war direkt an projected
    index = distances.argmin()

    # END SOLUTION

    return index
#

recognize_face(img, PC, MB_matrix_mean, asparagus_space)
