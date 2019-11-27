from grid import submit_script
import argparse
from mnist_keras_local
import train_keras
import os

if __name__ == "__main__":
        print(sys.argv)

        parser = argparse.ArgumentParser()
        parser.add_argument("mode",type=str, default="grid", choices=["grid","local"], help="Specify if the script shall run local for testing/debugging or in the grid")
        args = parser.parse_args()

        environment = 'source /net/projects/scratch/winter/valid_until_31_July_2020/mgerstenberg/asparagus/code/grid_framework/test_environments/virtualenv/bin/activate'
        #We run the script with different parameters either locally or in the grid
        for n_neurons in range(100,500,100):
            if args.mode == "local":
                train_keras(*[n_neurons, "adam",1])# Asterix used to unpack values from list.
            elif args.mode == "grid":
                #Mind that you can use the very same list of arguments as before ([n_neurons, "adam",1]). Submit gridjob with more than zero cuda cores if GPU computation is required.
                submit_script(os.getcwd()+"/mnist_keras.py",[n_neurons, "adam",1], environment)
