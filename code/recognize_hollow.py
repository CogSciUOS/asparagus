'''this is the recognize asparagus function
first done for hollow
'''

from scipy.spatial.distance import cdist
import cv2
from grid import*
from submit_recognize import*
import glob
import numpy as np
import matplotlib.pyplot as plt

img_shape = (1340,364,3)

def recognize(input, eigenasparagus, mean_asparagus, asparagus_space):
    """
    Recognize a face from a face database.
    and return the index of the best matching database entry.

    The FACE is first centered and projected into the eigeface
    space provided by EIGENFACES. Then the best match is found
    according to the euclidean distance in the eigenface space.

    Args:
        input (ndarray): asparagus to be recognised.
        eigenasparagus (ndarray): Array of eigenasparagus.
        mean_asparagus (ndarray): Average asparagus.
        asparagus_space (ndarray): Database of asparagus projectected into Eigenface space.

    Returns:
        index (uint): Position of the best matching face in face_db.
    """
    index = -1

    # BEGIN SOLUTION
    # center the face
    centered = input - mean_asparagus

    # and project it into the eigenface space
    projected = eigenasparagus @ centered

    # Now compute the similarity to all known faces
    # (comparison is performed in the eigenface space)
    distances = cdist(asparagus_space, projected[None, :])
    index = distances.argmin()

    # END SOLUTION
    print(index)
    return index

'''this function turns the index found into the binary class
if index is <199, argument is true
else, argument is not true
e.g. for hollow:
first 200 pictures in m_hollow are hollow
the others are not hollow
so if one of the pictures is recognized, lower than the index 200
the asparagus is hollow'''

def find_integer(index):
    n_index = 400
    if index <= 199:
        n_index = 0
    else:
        n_index = 1
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
    print('imgs[0] \n', imgs[0])
    print('neue Image_shape: \n', img_shape)
    plt.figure(figsize=(12, 12))
    plt.suptitle('Asparagus recognition based on {} principal components'.format(num_eigenfaces))
    plt.gray()
    for j, img in enumerate(imgs):

        # find the best match in the eigenface database
        winner = recognize(img.reshape(np.prod(img_shape)), path_to_PC, path_to_m_std, path_to_space)
        winner = find_integer(winner)
        name_label = labels[j][5:7]
        name_winner = train_labels[winner][5:7]

        plt.subplot(5, 8, 2 * j + 1)
        plt.axis('off')
        plt.imshow(img)
        plt.title(labels[j][5:7])

        plt.subplot(5, 8, 2 * j + 2)
        plt.axis('off')
        plt.imshow(train_imgs[winner])
        plt.title(('*' if name_label != name_winner else '') + name_winner)
    plt.show()


if __name__ == '__main__':
    args = typecast(sys.argv[1:])
    path_to_input = args[0]
    path_to_PC = args[1]
    path_to_m_std = args[2]
    path_to_space = args[3]
    path_to_eigenasparagus = args[4]
    path_to_m = args[5]

    #train_names = [200*hollow and 200* not_hollow]
    labels = ['hollow', 'not_hollow']
    num_eigenvectors = 4
    #read in some test data
    #test_img = np.zeros((10, img_shape[0]*img_shape[1]*img_shape[2]))
    test_img = np.zeros((10, img_shape[0],img_shape[1],img_shape[2]))
    s = 1
    for i in range(10):
        img = cv2.imread(path_to_input+'13644'+str(s+i)+'_b.png')
        test_img[i,:,:,:] = img
        #print(raw_ims.shape)
    print(test_img.shape)

    train_names_1 = ["hollow" for x in range(200)]
    train_names_2 = ["unhollow" for x in range(200)]
    train_names = np.concatenate((train_names_1,train_names_2), axis = 0)
    print(train_names.shape)


    show_recognition_results(test_img, labels, path_to_m, train_names, num_eigenvectors, path_to_eigenasparagus, path_to_m_std, path_to_space)
