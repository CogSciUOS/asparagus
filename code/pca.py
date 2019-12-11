from IPython.display import Image, display
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import cv2
import numpy as np
import matplotlib.gridspec as grid


img = cv2.imread('C:/Users/schmuri/github/asparagus/images/test_pca/G01-190520-162409-065_F00.bmp')
img_shape = img.shape[:2]
print('image size = ',img_shape)

# specify no of bands in the image
n_bands = 3

# 3 dimensional dummy array with zeros
MB_img = np.zeros((img_shape[0],img_shape[1],n_bands))

# stacking up images into the array
for i in range(n_bands):
    MB_img[:,:,i] = cv2.imread('band'+str(i+1)+'.jpg', cv2.IMREAD_GRAYSCALE)

# take a look at the asparagus
print('\n\nDispalying colour image of the scene')
plt.figure(figsize=(img_shape[0]/100,img_shape[1]/100)) #image size =  (1376, 1040)
plt.imshow(img, vmin=0, vmax=255)
plt.axis('off')
plt.show()

fig,axes = plt.subplots(2,4,figsize=(50,23),sharex='all', sharey='all')
fig.subplots_adjust(wspace=0.1, hspace=0.15)
#fig.suptitle('Intensities at Different Bandwidth in visible and Infra-red spectrum', fontsize=30)
axes = axes.ravel()
for i in range(n_bands):
    axes[i].imshow(MB_img[:,:,i], cmap='gray', vmin=0, vmax=255)
    #axes[i].set_title('band '+str(i+1),fontsize=25)
    #axes[i].axis('off')
fig.delaxes(axes[-1])
plt.show(axes.all())

#
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
