import numpy as np
import matplotlib.pyplot as plt
from skimage import io,transform
from skimage.transform import resize
from numpy import linalg as LA
import csv
import cv2

#Using 6 sample images for the first try

#['0_a.png', '0_b.png','0_c.png','1_a.png', '1_b.png','1_c.png']
#imlist = (io.imread_collection('/net/projects/scratch/summer/valid_until_31_January_2020/asparagus/Images/labled/kappa_images/2B/*.png'))
imlist = (io.imread_collection('ex_images/*.png'))

img_shape = imlist[0].shape
# this is our image size (1376, 1040)


#turn the image matrix of m x n x 3 to lists of rgb values i.e. (m*n) x 3.
# initializing with zeros.
res = []

for i in range(len(imlist)):
    print(i)
    #0
    # Using the skimage.transform function-- resize image (m x n x dim)
    #m = transform.resize(imlist[i],(1340, 364,3))
 # Reshape the matrix to a list of rgb values.
    arr = np.ravel(imlist[i], order='C')

    print(arr.shape)
    #(1463280,)

 # concatenate the vectors for every image with the existing list.

    res.append(arr)
    print(res[0].shape)
    print(len(res))
    # length x width x 3 : (1463280,)
    #write_to_file("image.csv", i)


# PCA computation
mean_face, eigen_vecs = cv2.PCACompute(res, mean = None)

# Generate eigenface from results
eigen_faces = []
for eigen_vec in eigen_vecs:
    face = eigen_vec.reshape(img_shape)
    eigen_faces.append(face)

# Display eigenfaces
plt.figure(figsize=(20,16))
plt.gray()
for i in range(20):
    plt.subplot(5, 4, i+1)
    plt.axis('off')
    plt.imshow(eigen_faces[i])
    plt.title('Eigenface {}'.format(i))
plt.show()



# #Subtract the mean
# # print list of vectors - 3 columns (rgb)
# m = res.mean(axis = 0)
# #print(m) [0.14694415 0.12558748 0.11998542] - means of 3 columns
#
# res = res - m
#
# #Calculate the covariance matrix.
# #data is 3 dimensional,the cov matrix will be 3x3
# R = np.cov(res, rowvar=False) #[[0.09184436 0.07866244 0.07482539]
#                               #[0.07866244 0.06747891 0.06427313]
#                               #[0.07482539 0.06427313 0.06147922]]
#
# #return eigenvalues and eigenvectors of covariance matrix: eigenvector with the highest value is also the principal component of the dataset
# evals, evecs = LA.eigh(R)
# idx = np.argsort(evals)[::-1]
# evecs = evecs[:,idx]
#
# # sort eigenvectors according to same index
# evals = evals[idx]
#
# # select the best 3 eigenvectors (3 is desired dimension
# # of rescaled data array)
# evecs = evecs[:, :3]
#
# #evecs30 = evecs[:, :30]
# #
# # # save best 30 evecs as binary data, so that we can check the size of all eigenvectors
# # np.save('evecs_all.npy', evecs30)
# # # human readable format
# # np.savetxt('evecs_all.txt', evecs30)
#
# # make a matrix with the three eigenvectors as its columns.
# evecs_mat = np.column_stack((evecs))
#
# # carry out the transformation on the data using eigenvectors
# # and return the re-scaled data, eigenvalues, and eigenvectors
# #give us the original data solely in terms of the components we chose.
# m_1a_anna = np.dot(evecs.T, res.T).T
#
# print(m_1a_anna.shape)
# np.save('2B_final.npy', m_1a_anna)
# # human readable
# np.savetxt('2B_final.txt', m_1a_anna)
#
#
# # save evecs_mat as binary data
# np.save('evecs_mata.npy', evecs_mat)
# # human readable
# np.savetxt('evecs_mata.txt', evecs_mat)
#
# # save evals as binary data
# np.save('evalsa.npy', evals)
# # human readable
# np.savetxt('evalsa.txt', evals)
#
#
#
# # Calling function for first image.
# # Re-scaling from 0-255 to 0-1.
# img = imlist[1]/255.0
# plt.imshow(img)
# plt.show()
# data_aug(img)
