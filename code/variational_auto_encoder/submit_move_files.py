from grid import*
from stack_images import*

if __name__ == '__main__':
    path = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/asparagus/code/variational_auto_encoder/move_files.py'
    path_to_imgs = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/with_background_pngs/'
    path_to_csv = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/asparagus/code/variational_auto_encoder/combined.csv'
    
    args = [path_to_imgs, path_to_csv]
   # env = 'source stack/bin/activate'

    submit_script(path, args)