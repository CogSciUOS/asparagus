import grid
from extract_features_local import *
import argparse

import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type = str, default = "grid", choices =["grid","local"])
    input_args = parser.parse_args()


    preprocessed_images = '/net/projects/scratch/summer/valid_until_31_January_2020/asparagus/preprocessed_images/with_background_rotated1/'
    path = os.path.join(os.getcwd(),"extract_features_local.py")
    env = "source /net/projects/scratch/winter/valid_until_31_July_2020/mgerstenberg/asparagus/code/advanced_feature_extraction/feature_env/bin/activate"
    args = [preprocessed_images,os.getcwd()]

    if input_args.mode == "local":
        extractor = AdvancedFeatureExtractor(args[0])#preprocessed_images
        extractor.generate_dataset(*args[1:])#outpath
    elif input_args.mode == "grid":
        submit_script(path,args,env)
