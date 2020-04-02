from grid import*

# remember here that normally path_to_imgs is labeled_with_background - I changed it to rotated for exploratory purpose

if __name__ == '__main__':
    path = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/malin/asparagus/code/pca_code/pca_processing/get_ids/feature_ids.py'
    path_to_imgs = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/labeled_rotated/'
    path_features = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/malin/asparagus/code/pca_code/combined_new.csv'
    args = [path_to_imgs, path_features]
    env = 'source /net/projects/scratch/winter/valid_until_31_July_2020/asparagus/sharedConda/bin/activate dataSet'

    submit_script(path, args, env)
