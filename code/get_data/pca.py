import numpy as np
import os
from grid import*
import sys
import cv2
import matplotlib.pyplot as plt
#from skimage.transform import rescale

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def get_files(PATH):
    # get all file paths from directory
    all_files = os.listdir(PATH)
    n = len(all_files)
    # save images in a list
    imgs = []
    for file_name in all_files[:3]:
        img = rgb2gray(plt.imread(PATH + file_name))
        print(img.shape)
        #dsize = (int(img.shape[0]*0.2), int(img.shape[1]*0.2))
        #img_rescaled = cv2.resize(img, dsize = dsize)
        imgs.append(img)
    return np.array(imgs)

def rescale_imgs(imgs):
    '''
    Take a list of images and rescale (flatten) them.
    Args: list of images
    Out:  list of rescaled images
    '''
    data = []
    for img in imgs:
        new = img.reshape((np.prod(img.shape)))
        data.append(new)
    return np.array(data)

def pca(data):
    """
    Perform principal component analysis.
    
    Args:
        data (ndarray): an array of shape (n,k),
        meaning n entries with k dimensions
        
    Returns: two arrays
        eigenvecs (ndarray): array of shape (k,k) holding the principal components in its columns.
        eigenvals (ndarray): k-vector holding the corresponding variances, in descending order.
    """
    data_centered = data - data.mean(axis=0)
    cov_matrix = np.cov(data_centered.T, rowvar=False)
    eigenvals, eigenvecs = np.linalg.eig(cov_matrix)
    # sort eigenvalues and vectors
    idx = np.argsort(eigenvals)[::-1]
    eigenvecs = eigenvecs[:,idx]
    eigenvals = eigenvals[idx]
    # compute principal components
    pc = np.dot(eigenvecs.T, data_centered)
    return eigenvecs, eigenvals, pc

def create_eigenspace(data, eigenvecs):
    '''
    Create eigenspace with the sample images and the given eigenvectors.
    Args: data contains the flattend images (n,k) n = number of images, k = image dimension
          eigenvecs contains the eigenvectors (l,k) l = number of iegenvectors chosen to use, k = image dimension
    Out:  asparagus_db = database with the projected asparagus
    '''
    data = data - data.mean(axis=0)
    asparagus_db = np.dot(data.T, eigenvecs)
    return asparagus_db

def best_match(img, db, eigenvecs, data):
    ''' 
    Project new asparagus to asparagus_db and find best match. 
    TODO: Try out different distance measures
    Args: img to be projected/classified
          db = asparagus_db = eigenspace
          eigenvecs to project the asparagus img
    Out:  best match found in the db by index, hopefully of the same class
    '''
    index = -1
    mean_asparagus = data.mean(axis=0)
    centered = img - mean_asparagus
    #project it into the eigenface space
    projected = np.dot(eigenvecs,centered)

    # Now compute the similarity to all known asparagus
    distances = cdist(db, projected[None, :])
    index = distances.argmin()

    return index

if __name__ == '__main__':
    args = typecast(sys.argv[1:])
    path = args[0]
    num_eigenvecs = args[1]
    imgs = get_files(path)
    data = rescale_imgs(imgs)
    img = data[0]
    print(img.shape)
    eigenvecs, eigenvals, pc = pca(data)
    print(eigenvals[:3])
    eigenvecs_used = eigenvecs[:num_eigenvecs]
    print(eigenvecs_used.shape)
    asparagus_db = create_eigenspace(data, eigenvecs_used)
    #best_match = best_match(img, asparagus_db, eigenvecs_used, data)
    #print(best_match)
    plt.imshow(pc[0].reshape(1340,364))
    plt.show()