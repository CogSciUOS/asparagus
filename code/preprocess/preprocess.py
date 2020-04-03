import os.path
import os
import csv
from get_valid_triples import get_valid_triples
from submit_script import submit_script
from preprocess_locally import *
import argparse

description = """
    Use for example:
    python preprocess.py --mode grid --root /net/projects/scratch/summer/valid_until_31_January_2020/asparagus/Images/unlabled/ 
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--mode', help='Run the program either in mode local or in mode grid', nargs="?")
    parser.add_argument('--root', help='This folder must contain .bmp files that match the naming convention. \n These files may reside in subfolders.)', nargs="?")
    parser.add_argument('--output_directory', help='Run the program either in mode local or in mode grid', nargs="?")

    mode = parser.parse_args().mode or 'local'
    root = parser.parse_args().root or "/net/projects/scratch/summer/valid_until_31_January_2020/asparagus/Images/labled/1A_Anna"
    outpath = parser.parse_args().output_directory or os.getcwd()+"/test"


    parser.add_argument('--ignore', help='Specify this string (ignore) if you want to ignore folders or files that contain ignore', nargs="?")
    ignore = parser.parse_args().ignore or "before2019"

    #safe list of valid names in a csv file; each row contains a triplet of file directories
    valid = get_valid_triples(root, ignore)
    with open(os.path.join(os.getcwd(), 'valid_files.csv'), 'w') as outfile:
        writer = csv.writer(outfile, delimiter=',')
        for i in valid:
            writer.writerow(i)

    if mode == "local":
        assert False
        perform_preprocessing(os.path.join(os.getcwd(),'valid_files.csv'), outpath, 0, -1)

    elif mode == "grid":
        #root = "/net/projects/scratch/summer/valid_until_31_January_2020/asparagus/Images/unlabled"
        #ignore = "before2019"

        n_pieces = len(valid)
        n_jobs = 5
        files_per_job = n_pieces // n_jobs
        for job in range(n_jobs):
            current_outpath = outpath + "/" + str(job)
            if job == n_jobs-1:
                args = [os.path.join(os.getcwd(),'valid_files.csv'), current_outpath, str(job*files_per_job), str(n_pieces)]
            else:
                args = [os.path.join(os.getcwd(),'valid_files.csv'), current_outpath, str(job*files_per_job), str(job*files_per_job+files_per_job-1)]
            submit_script(os.getcwd()+"/preprocess_locally.py",args)

        """

        args = [os.path.join(os.getcwd(),'valid_files.csv'), os.getcwd()+"/test", 0, -1]
        args = [str(x) for x in args]
        submit_script(os.path.join(os.getcwd(),"preprocess_locally"))"""

    #"/net/projects/scratch/summer/valid_until_31_January_2020/asparagus/Images/unlabled/"


