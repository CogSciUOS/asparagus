from grid import*


if __name__ == '__main__':
    path = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/maren/asparagus/code/recognize_hollow.py'
    path_to_input =  '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/labeled_with_background/'# for testing (test_img)
    path_to_PC = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/PC_hollow.npy'
    path_to_m_std = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/m_hollow_std.npy'
    path_to_space = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/m_hollow_space.npy'
    path_to_eigenasparagus = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/eig_hollow_used.npy'
    path_to_m = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/m_hollow.npy'
    args = [path_to_input, path_to_PC, path_to_m_std, path_to_space, path_to_eigenasparagus, path_to_m]
    #environment = 'source activate dataSet'
    environment = 'source /net/projects/scratch/winter/valid_until_31_July_2020/asparagus/sharedConda/bin/activate dataSet'
    #env = 'source stack/bin/activate'

    submit_script(path, args, environment)
