'''this is the recognize asparagus function
first done for hollow
'''

''' then bended'''

''' then blume'''

''' then rust head'''

''' then rust body'''

''' then violet'''

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

    # center the face
    print('input_shapr \n', input.shape) #(1463280,)
    mean_asparagus = mean_asparagus.mean(axis = 0)
    print('mean asps \n', mean_asparagus.shape)
    centered = input - mean_asparagus
    print('centered shape: \n', centered.shape) #(1463280,)
    print('eigenasparagus \n', eigenasparagus.shape) #(4, 1463280)

    # and project it into the eigenface space
    projected = eigenasparagus @ centered
    print(projected.shape) #(4,)
    print(asparagus_space.shape)# (400,4)

    # Now compute the similarity to all known faces
    # (comparison is performed in the eigenface space)
    distances = cdist(asparagus_space, projected[None,:])
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
    plt.figure(figsize=(20, 20))
    plt.suptitle('Asparagus recognition based on {} principal components'.format(num_eigenfaces))
    #plt.gray()
    for j, img in enumerate(imgs):

        # find the best match in the eigenface database
        winner = recognize(img.reshape(np.prod(img_shape)), path_to_eigenasparagus, path_to_m_std, path_to_space)
        #winner1 = find_integer(winner)
        name_label = labels[j]#[5:7]
        name_winner = train_labels[winner]#[5:7]

        plt.subplot(5, 4, 2 * j + 1)
        plt.axis('off')
        plt.imshow(img)
        plt.title(labels[j][5:7])
        #plt.show()

        plt.subplot(5, 4, 2 * j + 2)
        plt.axis('off')
        #ich glaube train_imgs müssen noch wieder gereshaped werden...
        plt.imshow(train_imgs[winner].reshape(img_shape))
        plt.title(('*' if name_label != name_winner else '') + name_winner)
        plt.show()
        #hollow
        plt.savefig('/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/recognize_hollow/recognize'+str(j)+'.png')
        #bended
        #plt.savefig('/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/recognize_bended/recognize'+str(j)+'.png')
        #blume
        #plt.savefig('/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/recognize_blume/recognize'+str(j)+'.png')
        #rust head
        #plt.savefig('/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/recognize_rust_head/recognize'+str(j)+'.png')
        #rust body
        #plt.savefig('/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/recognize_rust_body/recognize'+str(j)+'.png')
        #violet
        #plt.savefig('/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/recognize_violet/recognize'+str(j)+'.png')


if __name__ == '__main__':
    args = typecast(sys.argv[1:])
    path_to_input = args[0]
    path_to_PC = np.load(args[1])
    path_to_m_std = np.load(args[2])
    path_to_space = np.load(args[3])
    path_to_eigenasparagus = np.load(args[4])
    path_to_m = np.load(args[5])

    #remember train_names = [200*hollow and 200* not_hollow]

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

    #for hollow
    train_names_1 = ["hollow" for x in range(200)]
    train_names_2 = ["not_hollow" for x in range(200)]
    train_names = train_names_1 + train_names_2
    test_labels = ['not_hollow', 'not_hollow', 'not_hollow', 'not_hollow', 'not_hollow', 'not_hollow', 'not_hollow', 'not_hollow', 'not_hollow', 'not_hollow']

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


    show_recognition_results(test_img, test_labels, path_to_m, train_names, num_eigenvectors, path_to_PC, path_to_m_std, path_to_space)
