import os
import pandas as pd
import argparse

description = """ Combines several labelfiles generated with the label app into one large csv.
Sample call (assuming files are saved in folder test_files):
python assemble_dataset ./test_files test_labels.csv
"""

def combine(infile_folder, outfile):
    """Combines csv files. Writes data to csv.
    Args:
        infile_folder: A folder that contains csv files
        outfile: Path to the output csv
    """
    files = os.listdir(infile_folder)

    dataframes = []
    for file in files:
        df = pd.read_csv(os.path.join(infile_folder, file), sep =";", header=0, index_col=0)
        dataframes.append(df)
    combined = pd.concat(dataframes, sort = False)

    combined.to_csv(os.path.join(os.getcwd(),outfile), sep = ";")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('infile_folder', help='Path to folder with csv files generated with label app')
    parser.add_argument('outfile', help='Path to csv that will contain combined data')
    args = parser.parse_args()

    combine(args.infile_folder,args.outfile)
