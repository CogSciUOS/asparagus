import numpy as np
import matplotlib.pyplot as plt
from skimage import io,transform


#Using 6 sample images for the first try
#['0_a.png', '0_b.png','0_c.png','1_a.png', '1_b.png','1_c.png']
imlist = (io.imread_collection('ex_images/*.png'))
# this is our image size (1376, 1040)

# transform images to have all the same size
for i in range(len(imlist)):
    # Using the skimage.transform function-- resize image (m x n x dim).
	m = transform.resize(imlist[i],(256,256,3))

io.imshow_collection(imlist)
print(imlist)
print(m)
#print(imlist[0].shape())


#turn the image matrix of m x n x 3 to lists of rgb values i.e. (m*n) x 3.
# initializing with zeros.
res = np.zeros(shape=(1,3))

for i in range(len(imlist)):
    m=transform.resize(imlist[i],(1376, 1040,3))
	# Reshape the matrix to a list of rgb values.
    arr=m.reshape((1376*1040),3)
    # concatenate the vectors for every image with the existing list.
    res = np.concatenate((res,arr),axis=0)

# delete initial zeros' row
res = np.delete(res, (0), axis=0)
# print list of vectors - 3 columns (rgb)
print(res)
