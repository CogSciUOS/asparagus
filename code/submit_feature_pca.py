from grid import*


if __name__ == '__main__':
    path = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/malin/asparagus/code/feature_pca.py'
    args = []
    #environment = 'source activate dataSet'
    environment = 'source net/projects/scratch/winter/valid_until_31_July_2020/asparagus/sharedConda/bin/activate'
    #env = 'source stack/bin/activate'

    submit_script(path, args, environment)
