from grid import *
import argparse
import subprocess

if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("mode",type=str, default="grid", choices=["grid","local"], help="Specify if the script shall run local for testing/debugging or in the grid")
        args = parser.parse_args()

        environment = 'source /net/projects/scratch/winter/valid_until_31_July_2020/mgerstenberg/asparagus/code/grid_framework/test_environments/virtualenv/bin/activate'
        path = "/net/projects/scratch/winter/valid_until_31_July_2020/mgerstenberg/asparagus/code/variational_auto_encoder/variational_autoencoder_local.py"
        if args.mode == "local":
            p1 = subprocess.Popen([environment])
            p1.wait()
            #from variational_autoencoder_local import main
            p2 = subprocess.Popen(["python " + path])
        else:
            submit_script(path,[],environment)
