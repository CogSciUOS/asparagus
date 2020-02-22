from grid import*


if __name__ == '__main__':
    path = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/maren/asparagus/code/pca_processing/feature_pca.py'
    path_to_matrix = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/data_length/m_length.npy'

    args = [path_to_matrix]
    #environment = 'source activate dataSet'
    environment = 'source /net/projects/scratch/winter/valid_until_31_July_2020/asparagus/sharedConda/bin/activate dataSet'
    #env = 'source stack/bin/activate'

    submit_script(path, args, environment)
