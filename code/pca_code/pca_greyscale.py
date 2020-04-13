from IPython.display import Image, display
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import cv2
import numpy as np
import matplotlib.gridspec as grid
import wx #package wx
from PIL import Image
 #PIL package

#nicht preprocessed
#img = cv2.imread('C:/Users/schmuri/github/asparagus/images/test_pca/G01-190520-162409-065_F00.bmp')
#preprocessed
img = cv2.imread('C:/Users/schmuri/github/asparagus/images/test_pca/109278_00.png')
print(img)
#print(type(img)) #<class 'numpy.ndarray'>
#print(img)
#print(img.shape) #(1376, 1040, 3)


img_shape = img.shape[:2]

#print('image size = ',img_shape)    #image size =  (1376, 1040)

# specify no of bands in the image
n_bands = 3 # irgendwie sind die n_bands nicht wirklich unterschiedlich, oder zu wenig unterschiedlich wir brauchen keine 7, weil

# 3 dimensional dummy array with zeros
MB_img = np.zeros((img_shape[0],img_shape[1],n_bands))  #(1376, 1040, 7)
s = 0
# stacking up images into the array
for i in range(n_bands):
    #MB_img[:,:,i] = cv2.imread('band'+str(i+1)+'.jpg', cv2.IMREAD_GRAYSCALE)
    #das hier bedeutet, dass er jeweils drei bilder hat, in unterschiedlichen farbdingern
    #reading in all 3 images of one asparagus (unprocessed)
    MB_img[:,:,i] = cv2.imread('C:/Users/schmuri/github/asparagus/images/test_pca/109278_0'+str(s+i)+'.png', cv2.IMREAD_GRAYSCALE)
print(MB_img) # hier sind alle nur NAN....

# take a look at the asparagus'print('\n\nDispalying colour image of the scene')
plt.figure(figsize=(img_shape[0]/100,img_shape[1]/100)) #image size =  (1376, 1040)
plt.imshow(img, vmin=0, vmax=255)
plt.axis('off')
#plt.show()     # das nervt


#hier werden die verschiedenen bands gezeigt. eigentlich sollte unter den bands mehr unterschiede sein.
fig,axes = plt.subplots(1,4,figsize=(50,23),sharex='all', sharey='all')
#fig,axes = plt.subplots(2,4,figsize=(50,23),sharex='all', sharey='all')
fig.subplots_adjust(wspace=0.1, hspace=0.15)
fig.suptitle('Same piece but different views', fontsize=20)
axes = axes.ravel()
for i in range(n_bands):
     axes[i].imshow(MB_img[:,:,i], cmap='gray', vmin=0, vmax=255)
     axes[i].set_title('band '+str(s+i),fontsize=12)
     axes[i].axis('off')
fig.delaxes(axes[-1])
plt.show(axes.all()) #
#
# #####this is Standardization
# # Convert 2d band array in 1-d to make them as feature vectors and Standardization
MB_matrix = np.zeros((MB_img[:,:,0].size,n_bands))
for i in range(n_bands):
    MB_array = MB_img[:,:,i].flatten()  # covert 2d to 1d array
    MB_arrayStd = (MB_array - MB_array.mean())/MB_array.std()
    MB_matrix[:,i] = MB_arrayStd
MB_matrix.shape;

print(MB_matrix)
 #
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
print(EigVec.shape)
#Projecting data on Eigen vector directions resulting to Principal Components
PC = np.matmul(MB_matrix,EigVec)   #cross product
print(PC)
# # Generate Paiplot for original data and transformed PCs
Bandnames = ['Band 1','Band 2','Band 3']
a = sns.pairplot(pd.DataFrame(MB_matrix,
               columns = Bandnames),
               diag_kind='kde',plot_kws={"s": 3})
a.fig.suptitle("Pair plot of Band images")
PCnames = ['PC 1','PC 2','PC 3']
b = sns.pairplot(pd.DataFrame(PC,
               columns = PCnames),
                diag_kind='kde',plot_kws={"s": 3})
b.fig.suptitle("Pair plot of PCs")
plt.show()

#Information Retained by Principal Components
plt.figure(figsize=(8,6))
plt.bar([1,2,3],EigVal/sum(EigVal)*100,align='center',width=0.4,
        tick_label = ['PC1','PC2','PC3'])
plt.ylabel('Variance (%)')
plt.title('Information retention')
plt.show()

#going back, to see what happend
# Rearranging 1-d arrays to 2-d arrays of image size
PC_2d = np.zeros((img_shape[0],img_shape[1],n_bands))
for i in range(n_bands):
    PC_2d[:,:,i] = PC[:,i].reshape(-1,img_shape[1])
# normalizing between 0 to 255
PC_2d_Norm = np.zeros((img_shape[0],img_shape[1],n_bands))
for i in range(n_bands):
    PC_2d_Norm[:,:,i] = cv2.normalize(PC_2d[:,:,i],
                    np.zeros(img_shape),0,255 ,cv2.NORM_MINMAX)

fig,axes = plt.subplots(1,4,figsize=(50,23),sharex='all',
                        sharey='all')
fig.subplots_adjust(wspace=0.1, hspace=0.15)
fig.suptitle('Intensities of Principal Components ', fontsize=30)
axes = axes.ravel()
for i in range(n_bands):
    axes[i].imshow(PC_2d_Norm[:,:,i],cmap='gray', vmin=0, vmax=255)
    axes[i].set_title('PC '+str(i+1),fontsize=25)
    axes[i].axis('off')
fig.delaxes(axes[-1])
plt.show()

# C:\Users\schmuri\Anaconda3\envs\pyqt\lib\site-packages\pandas\core\dtypes\cast.py:729: ComplexWarning: Casting complex values to real discards the imaginary part

# # put 3 images of one asparagus piece into one list
# def preprocess(triple,outpath,file_id):
#     fpath1,fpath2,fpath3 = triple
#
#     os.makedirs(outpath, exist_ok=True)#Make dir if non existant
#
#
#     imgs = []# open images
#     try:
#         imgs.append(Image.open(fpath1))
#         imgs.append(Image.open(fpath2))
#         imgs.append(Image.open(fpath3))
#         assert len(imgs) == 3
#         assert len(list(np.array(imgs[0]).shape)) == 3
#         assert len(list(np.array(imgs[1]).shape)) == 3
#         assert len(list(np.array(imgs[2]).shape)) == 3
#     except Exception as e:
#         print("Could not load all images correctly. Triple:")
#         print(triple)
#         print(e)
#         return
#
#         outpaths = [outpath+str(file_id)+"_a.png",outpath+str(file_id)+"_b.png",outpath+str(file_id)+"_c.png"]
#
#
#
# # Initialize the algorithm and set the number of PC's
# #pca = PCA(n_components=2)
#
# # Fit the model to data
# #pca.fit(data)
# # Get list of PC's
# #pca.components_
# # Transform the model to data
# #pca.transform(data)
# # Get the eigenvalues
# #pca.explained_variance_ratio
#
#
#
# if __name__ == "__main__":
#     # to start with the submit script: define arguments
#     path_to_valid_names = sys.argv[1] #contains initfile (filenames)
#     outpath = sys.argv[2] #must contain a slash at the end
#     start_idx = int(sys.argv[3])
#     stop_idx = int(sys.argv[4])
#     path_to_valid_names = "/net/projects/scratch/summer/valid_until_31_January_2020/asparagus/Images/unlabled/valid_files.csv"
#     # get valid file names
#     valid_triples = []
#     with open(path_to_valid_names, 'r') as f:
#         reader = csv.reader(f)
#         # only read in the non empty lists
#         for row in filter(None, reader):
#             valid_triples.append(row)
#             print(valid_triples)
#
#     file_id = start_idx
#     files_per_folder = 10000
#
# for triple in valid_triples[start_idx:stop_idx]:
#     out = outpath
#     preprocess(triple,out,file_id)
#     file_id += 1
