from grid import*


if __name__ == '__main__':
    path = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/malin/asparagus/code/feature_pca.py'
    args = []
    environment = 'source activate dataSet'
    #env = 'source stack/bin/activate'

    submit_script(path, args)
