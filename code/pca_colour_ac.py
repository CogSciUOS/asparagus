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

raw_ims = np.zeros((n_bands,img_shape[0],img_shape[1],img_shape[2]))

for i in range(n_bands):
    img = cv2.imread('C:/Users/schmuri/github/asparagus/code/pca_images_all_classes/'+str(s+i)+'_b.png')
    raw_ims[i,:,:,:] = img
    #print(raw_ims.shape)
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
    #face = np.reshape(img,newshape = (img_shape[0],img.shape[1]*img.shape[2]))
    #face_img = face #np.zeros((img_shape[0],img.shape[1]*img.shape[2]))  #(1376, 1040)
    #face_img = face_img.flatten()
    #print(face_img.shape)


    #centered = face_img - MB_matrix_mean
    centered = face - MB_matrix_mean

    #centered = centered.flatten()
    print(centered.shape)#(1340, 1092),durch flatten (1463280,)
    print(eigenfaces.shape) #(1463280, 130)

    # and project it into the eigenface space
    projected = np.matmul(centered, eigenfaces)
    #print(projected.shape) #(130,)

    # Now compute the similarity to all known faces
    # (comparison is performed in the eigenface space)
    #projected = projected.T #diese beiden operationen brauchte ich, wenn ich
    #face_db = face_db.T    #die funktion auf nur ein bild angewand habe, jetzt mit der großen funktion scheint es überflüssig
    #print('jetzt mit großem input face_db: \n',face_db.shape) #(1463280, 4)(130, 4)
    #print('jetzt mit großem input projected: \n',projected.shape) #(1463280, 4)(130, 4)

    distances = cdist(face_db, projected[None, :])#[None, :] das war direkt an projected
    index = np.argmin(distances)
    #return index
#    index = int(round(index /10))
    #intex = int(index)
    print('index: \n',index)
    # END SOLUTION

    return index

#recognize_face(img, PC, MB_matrix_mean, asparagus_space)



def find_integer(index):
    n_index = 130
    if index <= 9:
        n_index = 0
    elif index <= 19:
        n_index = 1
    elif index <= 29:
        n_index = 2
    elif index <= 39:
        n_index = 3
    elif index <= 49:
        n_index = 4
    elif index <= 59:
        n_index = 5
    elif index <= 69:
        n_index = 6
    elif index <= 79:
        n_index = 7
    elif index <= 89:
        n_index = 8
    elif index <= 99:
        n_index = 9
    elif index <= 109:
        n_index = 10
    elif index <= 119:
        n_index = 11
    else:
        n_index  = 12
    print('n_index: \n', n_index)
    return n_index


# ... and now check your function on the training set ...
# BEGIN SOLUTION
def show_recognition_results(imgs, labels, train_imgs, train_labels,
                             num_eigenfaces, eigenfaces, mean_face, face_db):
    """Iterate over all face images and compute the best matching face in face_db.

    Args:
        imgs (list): List of test faces.
        train_imgs (list): List of training faces.
        train_labels (list): List of training labels.
        num_eigenfaces (uint): Number of eigenfaces.
        eigenfaces (list): List of the eigenfaces.
        mean_face (ndarray): Average face.
        face_db (ndarray): Database of faces projectected into Eigenface space.

    Returns:

    """

    img_shape = imgs[0].shape
    print('neue Image_shape: \n', img_shape)
    plt.figure(figsize=(45, 25))
    plt.suptitle(
        'Asparagus recognition based on {} principal components'.format(num_eigenfaces))
    for j, img in enumerate(imgs):

        # find the best match in the eigenface database
        winner = recognize_face(img.reshape(np.prod(img_shape)), eigenfaces, mean_face, face_db)
        #winner = find_integer(winner) # hier wird der index gereshapet und das funktioniert

        #das original
        #name_label = labels[j]
        #name_winner = train_labels[winner] # hier wird der index aus train_labels (dem großen) rausgesucht.
        name_label = labels[j]
        name_winner = train_labels[winner]


        plt.subplot(5, 8, 2 * j + 1)
        plt.axis('off')
        #img = train_imgs[j].reshape(img_shape)
        plt.imshow(img)
        print(labels[j])
        plt.title(labels[j], fontsize = 8)

        plt.subplot(5, 8, 2 * j + 2)
        plt.axis('off')

        plt.imshow(train_imgs[winner])
        plt.title(('*' if name_label != name_winner else '') + name_winner, fontsize = 8)
    plt.show()

#taking every tenth picture of MB_Matrix, to test our recognition  14196
train_imgs = np.zeros((13,img_shape[0],img_shape[1],img_shape[2]))
train_imgs = raw_ims[0:130:10,:,:,:]
print(train_imgs.shape)
train_names = ['Köpfe','Köpfe','Köpfe','Köpfe','Köpfe','Köpfe','Köpfe','Köpfe','Köpfe','Köpfe',
'Bona','Bona','Bona','Bona','Bona','Bona','Bona','Bona','Bona','Bona',
'Clara','Clara','Clara','Clara','Clara','Clara','Clara','Clara','Clara','Clara','Clara',
'Krumme','Krumme','Krumme','Krumme','Krumme','Krumme','Krumme','Krumme','Krumme','Krumme',
'violet','violet','violet','violet','violet','violet','violet','violet','violet','violet',
'2a','2a','2a','2a','2a','2a','2a','2a','2a','2a',
'2b','2b','2b','2b','2b','2b','2b','2b','2b','2b',
'Blume','Blume','Blume','Blume','Blume','Blume','Blume','Blume','Blume', 'Blume',
'dicke','dicke','dicke','dicke','dicke','dicke','dicke','dicke','dicke','dicke',
'hohle','hohle','hohle','hohle','hohle','hohle','hohle','hohle','hohle','hohle',
'rost','rost','rost','rost','rost','rost','rost','rost','rost','rost',
'suppe','suppe','suppe','suppe','suppe','suppe','suppe','suppe','suppe','suppe',
'anna','anna','anna','anna','anna','anna','anna','anna','anna','anna']
train_names1 = ['Köpfe','Bona','Clara','Krumme','violet','2a','2b','Blume','dicke','hohle', 'rost','suppe', 'anna']
show_recognition_results(train_imgs, train_names1, raw_ims, train_names, num_eigenvectors, eig_asparagus_used, MB_matrix_mean, asparagus_space)
