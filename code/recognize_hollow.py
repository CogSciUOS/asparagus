'''this is the recognize asparagus function
first done for hollow
'''

from scipy.spatial.distance import cdist

img_shape = (1376, 1040, 3)

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
    elif:
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
    plt.figure(figsize=(12, 12))
    plt.suptitle(
        'Face recognition based on {} principal components'.format(num_eigenfaces))
    plt.gray()
    for j, img in enumerate(imgs):

        # find the best match in the eigenface database
        winner = recognize_face(
            img.reshape(np.prod(img_shape)), eigenfaces, mean_face, face_db)
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

#train_names = [200*hollow and 200* not_hollow]
labels = ['hollow', 'not_hollow']

regocnize(test_img, PC_hollow, m_hollow_std, hollow_space)
show_recognition_results(train_imgs, labels, raw_ims, train_names, num_eigenvectors, eig_asparagus_used, MB_matrix_mean, asparagus_space)