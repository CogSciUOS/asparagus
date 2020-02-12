from grid import*
from stack_images import*

if __name__ == '__main__':
    path = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/asparagus/code/get_data/stack_images.py'
    path_to_imgs = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/labeled_with_background'
    path_to_save = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/stacked_horizontal/'
    args = [path_to_imgs, path_to_save]
   # env = 'source stack/bin/activate'

    submit_script(path, args)