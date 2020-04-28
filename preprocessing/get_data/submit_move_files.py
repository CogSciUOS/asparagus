from grid import*
from move_files import*

if __name__ == '__main__':
    path = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/sophia/asparagus/code/get_data/move_files.py'
    path_to_imgs = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/with_background_rotated1/'
    path_to_csv = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/sophia/asparagus/code/get_data/combined_new.csv'
    path_to_save = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/labeled_rotated'
    
    args = [path_to_imgs, path_to_csv, path_to_save]
    environment = 'source /net/projects/scratch/winter/valid_until_31_July_2020/asparagus/sharedConda/bin/activate MultiLabel'

    submit_script(path, args, environment)