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

#get the files (files contai 10 images of each class)

img = cv2.imread('C:/Users/schmuri/github/asparagus/images/test_pca/109278_00.png')

img_shape = img.shape[:2] #(1376, 1040, 3)

n_bands = 130 #weil ich 130 bilder habe

MB_img = np.zeros((img_shape[0],img_shape[1],n_bands))  #(1376, 1040, 7)
s = 0
# stacking up images into the array

for i in range(n_bands):
    #MB_img[:,:,i] = cv2.imread('band'+str(i+1)+'.jpg', cv2.IMREAD_GRAYSCALE)
    #das hier bedeutet, dass er jeweils drei bilder hat, in unterschiedlichen farbdingern
    #reading in all 3 images of one asparagus (unprocessed)
    #wie komme ich jetzt auf diesen path...'C:/Users/schmuri/github/asparagus/code/pca_images_all_classes/*.png'
    MB_img[:,:,i] = cv2.imread('C:/Users/schmuri/github/asparagus/code/pca_images_all_classes/'+str(s+i)+'_b.png', cv2.IMREAD_GRAYSCALE)

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
MB_matrix.shape;

print(MB_matrix)

#Compute eigenvectors and values
# Covariance
np.set_printoptions(precision=3)
cov = np.cov(MB_matrix.transpose())
# Eigen Values
EigVal,EigVec = np.linalg.eig(cov)
print("Eigenvalues:\n\n", EigVal,"\n")
  # Ordering Eigen values and vectors
order = EigVal.argsort()[::-1]
EigVal = EigVal[order]
EigVec = EigVec[:,order]
#Projecting data on Eigen vector directions resulting to Principal Components
PC = np.matmul(MB_matrix,EigVec)   #cross product
print(PC)

x = range(10)#np.linspace(0,130, 1)
plt.plot(x,EigVal[:10])
plt.show()
#save eigenvalues
#np.save('EigVal.npy', EigVal)
# human readable
#np.savetxt('evecs_mata.txt', evecs_mat)
