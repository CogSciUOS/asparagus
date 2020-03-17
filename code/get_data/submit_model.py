from grid import*
from multilabel_model1 import*

if __name__ == '__main__':
    path = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/sophia/asparagus/code/get_data/multilabel_model1.py'
    path_to_imgs = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/stacked_no_bg_horizontal/0/data_horizontal_noB.npy'
    path_to_labels = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/sophia/asparagus/code/get_data/combined_new.csv'
    args = [path_to_imgs, path_to_labels]
    environment = 'source /net/projects/scratch/winter/valid_until_31_July_2020/asparagus/sharedConda/bin/activate MultiLabel'

    submit_script(path, args, environment)