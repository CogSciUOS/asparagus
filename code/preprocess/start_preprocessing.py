import os.path
import os
import csv
from get_valid_triples import get_valid_triples
from submit_script import submit_script
from perform_preprocessing import *
import argparse

description = """

"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('mode', help='Run the program either in mode local or in mode grid', nargs="?")
    parser.add_argument('root', help='This folder must contain .bmp files that match the naming convention. \n These files may reside in subfolders.)', nargs="?")
    parser.add_argument('output_directory', help='Run the program either in mode local or in mode grid', nargs="?")
    outfiletype = "jpg"

    mode = parser.parse_args().mode or 'local'
    outpath = parser.parse_args().output_directory or os.getcwd()+"/test"
    root = parser.parse_args().root or "/net/projects/scratch/summer/valid_until_31_January_2020/asparagus/Images/labled/1A_Anna"

    parser.add_argument('ignore', help='Specify this string (ignore) if you want to ignore folders or files that contain ignore', nargs="?")
    ignore = parser.parse_args().ignore or ""

    #safe list of valid names in a csv file; each row contains a triplet of file directories
    valid = get_valid_triples(root, ignore)
    with open(os.path.join(os.getcwd(), 'valid_files.csv'), 'w') as outfile:
        writer = csv.writer(outfile, delimiter=',')
        for i in valid:
            writer.writerow(i)

    if mode == "local":
        perform_preprocessing(os.path.join(os.getcwd(),'valid_files.csv'), outpath, 0, -1, outfiletype)

    elif mode == "grid" or mode == "pseudogrid":
        #root = "/net/projects/scratch/summer/valid_until_31_January_2020/asparagus/Images/unlabled"
        #ignore = "before2019"

        n_pieces = len(valid)
        n_jobs = 5
        files_per_job = n_pieces // n_jobs
        for job in range(n_jobs):
            current_outpath = outpath + "/" + str(job)
            if job == n_jobs-1:
                args = [os.path.join(os.getcwd(),'valid_files.csv'), current_outpath, job*files_per_job, n_pieces,outfiletype]
            else:
                args = [os.path.join(os.getcwd(),'valid_files.csv'), current_outpath, job*files_per_job, (job*files_per_job+files_per_job)-1,outfiletype]
           
            if mode == "grid":
                submit_script(os.getcwd()+"/perform_preprocessing.py",args)
            elif mode == "pseudogrid":
                perform_preprocessing(*args)

        """

        args = [os.path.join(os.getcwd(),'valid_files.csv'), os.getcwd()+"/test", 0, -1]
        args = [str(x) for x in args]
        submit_script(os.path.join(os.getcwd(),"preprocess_locally"))"""

    #"/net/projects/scratch/summer/valid_until_31_January_2020/asparagus/Images/unlabled/"


