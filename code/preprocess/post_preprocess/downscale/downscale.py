from grid import *
import argparse
from downscale_local import *

if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("mode",type=str, default="grid", choices=["grid","local"], help="Specify if the script shall run local for testing/debugging or in the grid")
        args = parser.parse_args()

        environment = 'source /net/projects/scratch/winter/valid_until_31_July_2020/mgerstenberg/asparagus/code/grid_framework/test_environments/virtualenv/bin/activate'
        path = "/net/projects/scratch/winter/valid_until_31_July_2020/mgerstenberg/asparagus/code/preprocess/post_preprocess/downscale/downscale_local.py"
        if args.mode == "local":
            from downscale_local import main
            main()
        else:
            submit_script(path,[],environment)
