from grid import*
from multilabel_model1 import*

if __name__ == '__main__':
    path = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/asparagus/code/get_data/multilabel_model1.py'
    path_to_imgs = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/stacked_horizontal/0/data_horizontal.npy'
    path_to_labels = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/asparagus/code/get_data/combined_new.csv'
    args = [path_to_imgs, path_to_labels]
   # env = 'source stack/bin/activate'

    submit_script(path, args)