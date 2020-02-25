from grid import*


if __name__ == '__main__':
    #hollow
    # path = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/maren/asparagus/code/pca_processing/recognize_hollow.py'
    # path_to_input =  '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/labeled_with_background/'# for testing (test_img)
    # path_to_PC = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/data_hollow/PC_hollow_1.npy'
    # path_to_m_std = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/data_hollow/m_hollow_std_1.npy'
    # path_to_space = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/data_hollow/m_hollow_space_1.npy'
    # path_to_eigenasparagus = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/data_hollow/eig_hollow_used.npy'
    # path_to_m = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/data_hollow/m_hollow.npy'
    # args = [path_to_input, path_to_PC, path_to_m_std, path_to_space, path_to_eigenasparagus, path_to_m]
    # #environment = 'source activate dataSet'
    # environment = 'source /net/projects/scratch/winter/valid_until_31_July_2020/asparagus/sharedConda/bin/activate dataSet'
    # #env = 'source stack/bin/activate'


    #bended
    path = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/malin/asparagus/code/pca_processing/recognize_hollow.py'
    path_to_input =  '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/labeled_with_background/'# for testing (test_img)
    path_to_PC = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/data_bended/PC_bended.npy'
    path_to_m_std = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/data_bended/m_bended_std.npy'
    path_to_space = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/data_bended/m_bended_space.npy'
    path_to_eigenasparagus = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/data_bended/eig_bended_used.npy'
    path_to_m = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/data_bended/m_bended.npy'
    args = [path_to_input, path_to_PC, path_to_m_std, path_to_space, path_to_eigenasparagus, path_to_m]
    #environment = 'source activate dataSet'
    environment = 'source /net/projects/scratch/winter/valid_until_31_July_2020/asparagus/sharedConda/bin/activate dataSet'
    #env = 'source stack/bin/activate'
    submit_script(path, args, environment)
