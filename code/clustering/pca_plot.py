import numpy as np 
import matplotlib.pyplot as plt
import os


def show_images(images, cols = 1, titles = None):
    """Display a list of images in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    fig.suptitle("First 10 Eigenasparagus bended", fontsize=15, fontweight='bold')
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()

def get_pc_files(PATH):
    '''
    Get all principal component files
    Args: PATH to files
    Out: List of all file directories
    '''
    all_files = []
    for files in os.listdir(PATH):
        print(files)
        if files.endswith(".png"):
            if files.startswith("pca"):
                all_files.append(os.path.join(path,files))

    return all_files

if __name__ == '__main__':
    path = "/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/data_bended/"
    files = get_pc_files(path)
    print(files)
    images = []
    for file in files:
        images.append(plt.imread(file))
    titles = ["Eigenasparagus 1","Eigenasparagus 2","Eigenasparagus 3","Eigenasparagus 4","Eigenasparagus 5","Eigenasparagus 6","Eigenasparagus 7","Eigenasparagus 8","Eigenasparagus 9","Eigenasparagus 10"]
    show_images(images, cols = 3, titles = titles)
    #plt.show(figure)
    plt.savefig("pc_bended")