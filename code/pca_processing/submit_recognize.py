from grid import*


if __name__ == '__main__':
    #hollow
    path = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/maren/asparagus/code/pca_processing/recognize_hollow.py'
    path_to_input =  '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/labeled_with_background/'# for testing (test_img)
    path_to_PC = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/data_hollow/PC_hollow_1.npy'
    path_to_m_std = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/data_hollow/m_hollow_std_1.npy'
    path_to_space = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/data_hollow/m_hollow_space_1.npy'
    path_to_eigenasparagus = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/data_hollow/eig_hollow_used.npy'
    path_to_m = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/data_hollow/m_hollow.npy'
    args = [path_to_input, path_to_PC, path_to_m_std, path_to_space, path_to_eigenasparagus, path_to_m]
    #environment = 'source activate dataSet'
    environment = 'source /net/projects/scratch/winter/valid_until_31_July_2020/asparagus/sharedConda/bin/activate dataSet'
    #env = 'source stack/bin/activate'


    # #bended
    # path = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/malin/asparagus/code/pca_processing/recognize_hollow.py'
    # path_to_input =  '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/labeled_with_background/'# for testing (test_img)
    # path_to_PC = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/data_bended/PC_bended.npy'
    # path_to_m_std = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/data_bended/m_bended_std.npy'
    # path_to_space = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/data_bended/m_bended_space.npy'
    # path_to_eigenasparagus = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/data_bended/eig_bended_used.npy'
    # path_to_m = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/data_bended/m_bended.npy'
    # args = [path_to_input, path_to_PC, path_to_m_std, path_to_space, path_to_eigenasparagus, path_to_m]
    # #environment = 'source activate dataSet'
    # environment = 'source /net/projects/scratch/winter/valid_until_31_July_2020/asparagus/sharedConda/bin/activate dataSet'
    # #env = 'source stack/bin/activate'


    #blume
    # path = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/malin/asparagus/code/pca_processing/recognize_hollow.py'
    # path_to_input =  '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/labeled_with_background/'# for testing (test_img)
    # path_to_PC = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/data_blume/PC_blume.npy'
    # path_to_m_std = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/data_blume/m_blume_std.npy'
    # path_to_space = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/data_blume/m_blume_space.npy'
    # path_to_eigenasparagus = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/data_blume/eig_blume_used.npy'
    # path_to_m = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/data_blume/m_blume.npy'
    # args = [path_to_input, path_to_PC, path_to_m_std, path_to_space, path_to_eigenasparagus, path_to_m]
    # environment = 'source /net/projects/scratch/winter/valid_until_31_July_2020/asparagus/sharedConda/bin/activate dataSet'


    #rost head - STILL NOT WORKING!
    # path = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/malin/asparagus/code/pca_processing/recognize_hollow.py'
    # path_to_input =  '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/labeled_with_background/'# for testing (test_img)
    # path_to_PC = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/data_rost_head/PC_rost_head.npy'
    # path_to_m_std = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/data_rost_head/m_rost_head_std.npy'
    # path_to_space = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/data_rost_head/m_rost_head_space.npy'
    # path_to_eigenasparagus = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/data_rost_head/eig_rost_head_used.npy'
    # path_to_m = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/data_rost_head/m_rost_head.npy'
    # args = [path_to_input, path_to_PC, path_to_m_std, path_to_space, path_to_eigenasparagus, path_to_m]
    # environment = 'source /net/projects/scratch/winter/valid_until_31_July_2020/asparagus/sharedConda/bin/activate dataSet'


    #rost body
    # path = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/malin/asparagus/code/pca_processing/recognize_hollow.py'
    # path_to_input =  '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/labeled_with_background/'# for testing (test_img)
    # path_to_PC = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/data_rost_body/PC_rost_body.npy'
    # path_to_m_std = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/data_rost_body/m_rost_body_std.npy'
    # path_to_space = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/data_rost_body/m_rost_body_space.npy'
    # path_to_eigenasparagus = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/data_rost_body/eig_rost_body_used.npy'
    # path_to_m = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/data_rost_body/m_rost_body.npy'
    # args = [path_to_input, path_to_PC, path_to_m_std, path_to_space, path_to_eigenasparagus, path_to_m]
    # environment = 'source /net/projects/scratch/winter/valid_until_31_July_2020/asparagus/sharedConda/bin/activate dataSet


    #violet
    # path = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/malin/asparagus/code/pca_processing/recognize_hollow.py'
    # path_to_input =  '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/labeled_with_background/'# for testing (test_img)
    # path_to_PC = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/data_violet/PC_violet.npy'
    # path_to_m_std = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/data_violet/m_violet_std.npy'
    # path_to_space = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/data_violet/m_violet_space.npy'
    # path_to_eigenasparagus = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/data_violet/eig_violet_used.npy'
    # path_to_m = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/data_violet/m_violet.npy'
    # args = [path_to_input, path_to_PC, path_to_m_std, path_to_space, path_to_eigenasparagus, path_to_m]
    # environment = 'source /net/projects/scratch/winter/valid_until_31_July_2020/asparagus/sharedConda/bin/activate dataSet


    submit_script(path, args, environment)
