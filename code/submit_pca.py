from grid import*


if __name__ == '__main__':
    path = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/maren/asparagus/code/feature_ids.py'
    path_to_imgs = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/labeled_with_background/'
    path_features = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/maren/asparagus/code/combined_new.csv'
    args = [path_to_imgs, path_features]
   # env = 'source stack/bin/activate'

    submit_script(path, args)
