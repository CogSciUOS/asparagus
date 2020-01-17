import numpy as np
import matplotlib.pyplot as plt
from skimage import io,transform
from skimage.transform import resize
from numpy import linalg as LA
import csv


# function for data augmentation
def data_aug(img):
    mu = 0
    sigma = 0.1
    feature_vec=np.matrix(evecs_mat)
	# 3 x 1 scaled eigenvalue matrix
    se = np.zeros((3, 1))
    se[0][0] = np.random.normal(mu, sigma)*evals[0]
    se[1][0] = np.random.normal(mu, sigma)*evals[1]
    se[2][0] = np.random.normal(mu, sigma)*evals[2]
    se = np.matrix(se)
    val = feature_vec*se
    #print(val.shape)

	# Parse through every pixel value.
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # Parse through every dimension.fail
            for k in range(img.shape[2]):
                img[i,j,k] = float(img[i,j,k]) + float(val[k])

#Using 6 sample images for the first try

#['0_a.png', '0_b.png','0_c.png','1_a.png', '1_b.png','1_c.png']
imlist = (io.imread_collection('/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/Images/labled/kappa_images/1A_Anna/*.png'))
#imlist = (io.imread_collection('ex_images/*.png'))
# this is our image size (1376, 1040)


#turn the image matrix of m x n x 3 to lists of rgb values i.e. (m*n) x 3.
# initializing with zeros.
res = np.zeros(shape=(1,3))

for i in range(len(imlist)):
    # Using the skimage.transform function-- resize image (m x n x dim)
    m = transform.resize(imlist[i],(1340, 364,3))
 # Reshape the matrix to a list of rgb values.
    arr = m.reshape((1340*364),3)
 # concatenate the vectors for every image with the existing list.
    res = np.concatenate((res,arr),axis=0)
    #write_to_file("image.csv", i)


# delete initial zeros' row
res = np.delete(res, (0), axis=0)

#Subtract the mean
# print list of vectors - 3 columns (rgb)
m = res.mean(axis = 0)
#print(m) [0.14694415 0.12558748 0.11998542] - means of 3 columns

res = res - m

#Calculate the covariance matrix.
#data is 3 dimensional,the cov matrix will be 3x3
R = np.cov(res, rowvar=False) #[[0.09184436 0.07866244 0.07482539]
                              #[0.07866244 0.06747891 0.06427313]
                              #[0.07482539 0.06427313 0.06147922]]

evals, evecs = LA.eigh(R) #return eigenvalues and eigenvectors
idx = np.argsort(evals)[::-1]
evecs = evecs[:,idx]

# sort eigenvectors according to same index
evals = evals[idx]

# select the best 3 eigenvectors (3 is desired dimension
# of rescaled data array)
evecs = evecs[:, :3]

# evecs30 = evecs[:, :30]
#
# # save best 30 evecs as binary data, so that we can check the size of all eigenvectors
# np.save('evecs_all.npy', evecs30)
# # human readable format
# np.savetxt('evecs_all.txt', evecs30)

# make a matrix with the three eigenvectors as its columns.
evecs_mat = np.column_stack((evecs))

# carry out the transformation on the data using eigenvectors
# and return the re-scaled data, eigenvalues, and eigenvectors
m = np.dot(evecs.T, res.T).T


# save evecs_mat as binary data
np.save('evecs_mat.npy', evecs_mat)
# human readable
np.savetxt('evecs_mat.txt', evecs_mat)

# save evals as binary data
np.save('evals.npy', evals)
# human readable
np.savetxt('evals.txt', evals)



# Calling function for first image.
# Re-scaling from 0-255 to 0-1.
img = imlist[1]/255.0
# plt.imshow(img)
# plt.show()
data_aug(img)
