import numpy as np
import matplotlib.pyplot as plt
from skimage import io,transform
from skimage.transform import resize


#Using 6 sample images for the first try
#['0_a.png', '0_b.png','0_c.png','1_a.png', '1_b.png','1_c.png']
imlist = (io.imread_collection('ex_images/*.png'))
# this is our image size (1376, 1040)

#mb = transform.resize(imlist, (1340, 364,3))
# transform images to have all the same size
#for i in range(len(imlist)):
    # Using the skimage.transform function-- resize image (m x n x dim).
#    m = transform.resize(imlist[i],(1340, 364,3))

print(imlist[0].shape)
print(type(imlist[0]))
#print(m)
#print(m) - ist alles nur mit 0 ist das ein problem?
#print(imlist[0].shape())


#turn the image matrix of m x n x 3 to lists of rgb values i.e. (m*n) x 3.
# initializing with zeros.
# res = np.zeros(shape=(1,3))
# #
# for i in range(len(imlist)):
#      #m = transform.resize(imlist[i],(1340, 364,3))
#  	# Reshape the matrix to a list of rgb values.
#      arr = imlist.reshape((1340*364),3)
#      # concatenate the vectors for every image with the existing list.
#      res = np.concatenate((res,arr),axis=0)
#
#  # delete initial zeros' row
# res = np.delete(res, (0), axis=0)
# # print list of vectors - 3 columns (rgb)
# print(res)
