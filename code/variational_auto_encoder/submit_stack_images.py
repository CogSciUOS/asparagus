from grid import*
from stack_images import*

if __name__ == '__main__':
    path = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/asparagus/code/variational_auto_encoder/stack_images.py'
    args = ['with_background_pngs/0/0', 'stacked_images/']
   # env = 'source stack/bin/activate'

    submit_script(path, args)