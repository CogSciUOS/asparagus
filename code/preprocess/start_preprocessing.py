from submit_script import *
import os
import csv

if __name__ == "__main__":
    root = "/net/projects/scratch/summer/valid_until_31_January_2020/asparagus/Images/labled/"
    csv_files = [f for f in os.listdir(root) if f.endswith('.csv')]
    classes = ["1A_Anna", "1A_Bona", "1A_Clara", "1A_Krumme", "1A_Violett", "2A", "2B", "Blume", "Dicke", "Hohle", "Köpfe", "Rost", "Suppe"]
    
    for csv_file, c in zip(csv_files, classes):
        path_to_valid_names = csv_file
        outpath = root + "/kappa_images/" + c + "/"
    
        files_per_job = 10000
	    # get valid file names
        valid_triples = []
        with open(path_to_valid_names, 'r') as f:
            reader = csv.reader(f)
		    # only read in the non empty lists
            for row in filter(None, reader):
                valid_triples.append(row)
        n_pieces = len(valid_triples)
        n_jobs = n_pieces // files_per_job
        for job in range(n_jobs):
            current_outpath = outpath
            if job == n_jobs-1:
                args = [path_to_valid_names, current_outpath, str(job*files_per_job), str(n_pieces)]
            else:
                args = [path_to_valid_names, current_outpath, str(job*files_per_job), str(job*files_per_job+files_per_job)]
            submit_script(os.getcwd()+"/preprocess.py",args)