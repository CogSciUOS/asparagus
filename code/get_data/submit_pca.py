from grid import*
from stack_images import*

if __name__ == '__main__':
    path = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/asparagus/code/get_data/pca.py'
    path_to_imgs = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/pca_images_all_classes/'
    num_eigenvecs = 10
    args = [path_to_imgs, num_eigenvecs]
   # env = 'source stack/bin/activate'

    submit_script(path, args)