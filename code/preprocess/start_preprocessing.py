import os.path
import os
import csv
from get_valid_triples import get_valid_triples
from submit_script import submit_script
from perform_preprocessing import *
import argparse
import sys

description = """Performs preprocessing either in the grid or locally.

If no valid_files.csv is present in the current working directory yet it is created by accumulating all triples of files in the root you provide with arg3.
Otherwise valid_files.csv is used.
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('mode', help='Run the program either in mode local or in mode grid', nargs="?")
    parser.add_argument('outfiletype', help='Filetype of the preprocessed images', nargs="?")
    parser.add_argument('root', help='This folder must contain .bmp files that match the naming convention. \n These files may reside in subfolders.)', nargs="?")
    parser.add_argument('output_directory', help='Run the program either in mode local or in mode grid', nargs="?")
    parser.add_argument('n_jobs',help="The number of gridjobs for preprocessing",nargs="?")
    parser.add_argument('with_background',help="Remove the background?",nargs="?")
    parser.add_argument('ignore', help='Specify this string (ignore) if you want to ignore folders or files that contain ignore', nargs="?")

    args = parser.parse_args()
    mode = args.mode or 'pseudogrid'
    outpath = args.output_directory or os.getcwd()+"/test"
    root = args.root or "/net/projects/scratch/summer/valid_until_31_January_2020/asparagus/Images/labled/1A_Anna"
    outfiletype = args.outfiletype or 'jpg'
    with_background = args.with_background or False
    with_background = int(with_background)
    ignore = parser.parse_args().ignore or ""

    print(root)
    #safe list of valid names in a csv file; each row contains a triplet of file directories
    valid = []
    if os.path.isfile(os.path.join(os.getcwd(), 'valid_files.csv')):
        print("Loading list of files from valid_files.csv")
        with open(os.path.join(os.getcwd(), 'valid_files.csv'), "r") as infile:
              reader = csv.reader(infile, delimiter=';')
              valid = list(reader)
    else:
        valid = get_valid_triples(root, ".bmp","before2019")
        with open(os.path.join(os.getcwd(), 'valid_files.csv'), 'w') as outfile:
            writer = csv.writer(outfile, delimiter=';')
            for i in valid:
                writer.writerow(i)

    #Ensure that valid files exist
    if type(valid) == type([]):
        if len(valid) == 0:
           print("No valid triples found")
           sys.exit(1)
    else:
        print("No valid triples")
        sys.exit(1)

    if mode == "local":
        perform_preprocessing(os.path.join(os.getcwd(),'valid_files.csv'), outpath, 0, -1, outfiletype, with_background)

    elif mode == "grid" or mode == "pseudogrid":
        #root = "/net/projects/scratch/summer/valid_until_31_January_2020/asparagus/Images/unlabled"
        #ignore = "before2019"

        n_pieces = len(valid)
        n_jobs = args.n_jobs or 5
        n_jobs = int(n_jobs)

        files_per_job = n_pieces // n_jobs
        for job in range(n_jobs):
            current_outpath = outpath + "/" + str(job)
            if job == n_jobs-1:
                args = [os.path.join(os.getcwd(),'valid_files.csv'), current_outpath, job*files_per_job, n_pieces,outfiletype,with_background]
            else:
                args = [os.path.join(os.getcwd(),'valid_files.csv'), current_outpath, job*files_per_job, (job*files_per_job+files_per_job)-1,outfiletype,with_background]
            if mode == "grid":
                submit_script(os.getcwd()+"/perform_preprocessing.py",args)
            elif mode == "pseudogrid":
                p = Preprocessor()
                p.perform_preprocessing(*args)

        """

        args = [os.path.join(os.getcwd(),'valid_files.csv'), os.getcwd()+"/test", 0, -1]
        args = [str(x) for x in args]
        submit_script(os.path.join(os.getcwd(),"preprocess_locally"))"""

    #"/net/projects/scratch/summer/valid_until_31_January_2020/asparagus/Images/unlabled/"
