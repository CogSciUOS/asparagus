from grid import*
from stack_images import*

if __name__ == '__main__':
    path = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/asparagus/code/variational_auto_encoder/combine_npyy.py'
    path_to_npy = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/stacked_labeled_with_background/0/'
    args = [path_to_npy]
   # env = 'source stack/bin/activate'

    submit_script(path, args)