'''this is the recognize asparagus function
first done for hollow, then bended, then bended, then rust_head,
then rust_body, then violet, then width and finally length
'''

from scipy.spatial.distance import cdist
import cv2
from grid import*
from submit_recognize import*
import glob
import numpy as np
import matplotlib.pyplot as plt

img_shape = (1340,364,3)

def classify(input, eigenasparagus, mean_asparagus, asparagus_space):
    """
    Recognizes an asparagus piece from a 400 pieces asparagus database
    and returns the index of the best matching piece of the database

    The asparagus piece is first centered and projected into the asparagus_space
    space provided by the eigen_asparagus. Then the best match is found
    according to the euclidean distance in the eigen_asparagus space.

    Args:
        input (ndarray): asparagus to be recognised.
        eigenasparagus (ndarray): Array of eigenasparagus.
        mean_asparagus (ndarray): Average asparagus.
        asparagus_space (ndarray): Database of asparagus projectected into asparagus space.

    Returns:
        index (uint): Position of the best matching asparagus in asparagus_space.
    """
    index = -1

    # center the piece
    mean_asparagus = mean_asparagus.mean(axis = 0)
    centered = input - mean_asparagus

    # project it into the asparagus space
    projected = eigenasparagus @ centered

    # compute the similarity to all known pieces
    # (comparison is performed in the asparagus_space)
    distances = cdist(asparagus_space, projected[None,:])
    index = distances.argmin()

    return index


def find_integer(index):

'''this function turns the index found into the binary class
if index is <199, argument is true
else, argument is not true
e.g. for hollow:
first 200 pictures in m_hollow are hollow
the others are not hollow
so if one of the pictures is recognized, lower than the index 200
the asparagus is classified as hollow'''

    n_index = 400
    if index <= 199:
        n_index = 0
    else:
        n_index = 1
    print('n_index: \n', n_index)
    return n_index


def show_classify_results(test_imgs, test_labels, train_imgs, train_labels,
                             num_eigenvectors, PC, mean_asparagus, asparagus_space):
    """Iterate over all asparagus images and compute the best matching asparagus in asparagus_space.

    Args:
        test_imgs (list): List of 10 test asparagus.
        test_labels(list): List of correct labels of test_imgs
        train_imgs (matrix): Matrix of 400 feature asparagus images.
        train_labels (list): Labels of the train_imgs (200 with feature, 200 without).
        num_eigenvectors (uint): Number of eigenvectors used.
        PC (matrix): The principle components.
        mean_asparagus (ndarray): Average asparagus.
        space (ndarray): Database of asparagus projectected into asparagus space.

    Returns:
    classification result (image) - score

    """
    img_shape = test_imgs[0].shape

    plt.figure(figsize=(20, 20))
    plt.suptitle('Asparagus classified based on {} principal components'.format(num_eigenvectors))

    for j, img in enumerate(test_imgs):

        # find the best match in the database
        winner = classify(img.reshape(np.prod(img_shape)), path_to_eigenasparagus, path_to_m_std, path_to_space)

        name_label = test_labels[j]#[5:7]
        name_winner = train_labels[winner]#[5:7]

        plt.subplot(5, 4, 2 * j + 1)
        plt.axis('off')
        plt.imshow(img)
        plt.title(test_labels[j][5:7])

        plt.subplot(5, 4, 2 * j + 2)
        plt.axis('off')
        plt.imshow(train_imgs[winner].reshape(img_shape))
        plt.title(('*' if name_label != name_winner else '') + name_winner)
        plt.show()
        plt.savefig('/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/recognize_length/recognize'+str(j)+'.png')


if __name__ == '__main__':
    args = typecast(sys.argv[1:])
    path_to_input = args[0]
    path_to_PC = np.load(args[1])
    path_to_m_std = np.load(args[2])
    path_to_space = np.load(args[3])
    path_to_eigenasparagus = np.load(args[4])
    path_to_m = np.load(args[5])

    num_eigenvectors = 4
    #read in test data
    test_imgs = np.zeros((10, img_shape[0],img_shape[1],img_shape[2]))
    s = 1
    for i in range(10):
        img = cv2.imread(path_to_input+'13644'+str(s+i)+'_b.png')
        test_imgs[i,:,:,:] = img
    print(test_imgs.shape)

    #for hollow
    # train_names_1 = ["hollow" for x in range(200)]
    # train_names_2 = ["not_hollow" for x in range(200)]
    # train_names = train_names_1 + train_names_2
    # test_labels = ['not_hollow', 'not_hollow', 'not_hollow', 'not_hollow', 'not_hollow', 'not_hollow', 'not_hollow', 'not_hollow', 'not_hollow', 'not_hollow']

    # # for bended
    # train_names_1 = ["bended" for x in range(200)]
    # train_names_2 = ["not_bended" for x in range(200)]
    # train_names = train_names_1 + train_names_2
    # test_labels = ['not_bended', 'bended', 'not_bended', 'bended', 'bended', 'bended', 'not_bended', 'bended', 'bended', 'not_bended']

    # for blume
    # train_names_1 = ["blume" for x in range(200)]
    # train_names_2 = ["not_blume" for x in range(200)]
    # train_names = train_names_1 + train_names_2
    # test_labels = ['not_blume', 'not_blume', 'blume', 'not_blume', 'not_blume', 'not_blume', 'not_blume','blume', 'not_blume', 'not_blume']

    # rost head
    # train_names_1 = ["rust_head" for x in range(200)]
    # train_names_2 = ["not_rust_head" for x in range(200)]
    # train_names = train_names_1 + train_names_2
    # test_labels = ['not_rust_head', 'not_rust_head', 'not_rust_head', 'not_rust_head', 'not_rust_head', 'not_rust_head', 'not_rust_head','not_rust_head', 'not_rust_head', 'rust_head']

    # rust body
    # train_names_1 = ["rust_body" for x in range(200)]
    # train_names_2 = ["not_rust_body" for x in range(200)]
    # train_names = train_names_1 + train_names_2
    # test_labels = ['not_rust_body', 'not_rust_body', 'not_rust_body', 'not_rust_body', 'not_rust_body', 'not_rust_body', 'not_rust_body','not_rust_body', 'not_rust_body', 'rust_body']

    #violet
    # train_names_1 = ["violet" for x in range(200)]
    # train_names_2 = ["not_violet" for x in range(200)]
    # train_names = train_names_1 + train_names_2
    # test_labels = ['not_violet', 'not_violet', 'not_violet', 'violet', 'not_violet', 'not_violet', 'not_violet','not_violet', 'not_violet', 'not_violet']

    #width (wider than 20 is width, other is not_width)
    # train_names_1 = ["width" for x in range(200)]
    # train_names_2 = ["not_width" for x in range(200)]
    # train_names = train_names_1 + train_names_2
    # test_labels = ['width', 'not_width', 'width', 'not_width', 'width', 'not_width', 'not_width','not_width', 'width', 'width']

    #length (longer than 210 mm is length, other is not_length)
    train_names_1 = ["length" for x in range(200)]
    train_names_2 = ["not_length" for x in range(200)]
    train_names = train_names_1 + train_names_2
    test_labels = ['length', 'length', 'length', 'length', 'length', 'length', 'length','length', 'length', 'length']


    show_classify_results(test_imgs, test_labels, path_to_m, train_names, num_eigenvectors, path_to_PC, path_to_m_std, path_to_space)
