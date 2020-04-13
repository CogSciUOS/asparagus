# 1. Description

The preprocessor collects paths for images with .bmp extension in the specified directory and all subdirectories, performs preprocessing steps locally or in the grid and saves the output in a nested structure in the output directory where each output file index is unique and follows the convention examplified by the following filenames: 1_a.png, 1_b.png, 1_c.png.

Use python start_preprocessing.py --help to get more information of how to use the preprocessor.

# 2. Sample usage

## 2.1 Submit gridjobs

The following example calls are used for preprocessing with/without background removal using five gridjobs:

python start_preprocessing.py grid png /net/projects/scratch/summer/valid_until_31_January_2020/asparagus/Images/unlabled /net/projects/scratch/summer/valid_until_31_January_2020/asparagus/preprocessed_images/with_background_pngs 5 1 before2019

python start_preprocessing.py grid png /net/projects/scratch/summer/valid_until_31_January_2020/asparagus/Images/unlabled /net/projects/scratch/summer/valid_until_31_January_2020/asparagus/preprocessed_images/without_background_pngs 5 0 before2019

## 2.2 Test in pseudogrid mode

The following calls were used for pseudogrid (local) jobs for testing

python start_preprocessing.py pseudogrid jpg /net/projects/scratch/summer/valid_until_31_January_2020/asparagus/Images/labled/1A_Anna /net/projects/scratch/summer/valid_until_31_January_2020/asparagus/preprocessed_images/test_anna 5 1 before2019

python start_preprocessing.py pseudogrid jpg /net/projects/scratch/summer/valid_until_31_January_2020/asparagus/Images/labled /net/projects/scratch/summer/valid_until_31_January_2020/asparagus/preprocessed_images/test_anna2 5 1 before2019

