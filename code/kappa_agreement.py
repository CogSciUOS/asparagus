""" This module computes the kappa agreement scores of our labelling outputs.
   Please provide 3 commandline arguments: 
   1) the first annotator csv file
   2) the second annotator csv file
   3) the outputfile name ("agreement" + first_name + second_name)

   Example: python kappa_agreement.py ../annotations/annotator_1.csv ../annotations/annotator_2.csv agreement_annotator_1_annotator_2.csv
"""

import sys
import csv

import pandas as pd
import numpy as np

from sklearn.metrics import cohen_kappa_score


def compute_agreement(filename_1, filename_2, threshold_score=0.8):
    """This function takes two csv files of annotations (with labels 1.0 and 0.0)
    and computes Cohen’s kappa, a score that expresses the level of agreement between two annotators on a classification problem.

    Arguments:
        filename_1 (str): first annotation file name
        filename_2 (str): first annotation file name

    Keyword Arguments:
        threshold_score (float): threshold whether agreement is acceptable or not (default: (0.8))

    Returns:
        a dictionary of {category: kappa_score}
    """

    # read in files
    annotations_1 = pd.read_csv(filename_1, delimiter=";")
    annotations_2 = pd.read_csv(filename_2, delimiter=";")

    # Check for NaNs
    if np.any(annotations_1.isna()) or np.any(annotations_2.isna()):
        raise ValueError(
            "There are missing values in the csv file. Fix it and come back here!")

    print()
    print("Scores above .8 are generally considered good agreement; zero or lower means no agreement (practically random labels)!")
    print()

    # save kappas in dict
    kappa_dict = {column: cohen_kappa_score(annotations_1[column], annotations_2[column], labels=[
        1.0, 0.0]) for column in annotations_1.columns[1:]}

    return kappa_dict


def write_to_file(filename, kappa_dict):
    """takes a dictionary of kappas and writes them into a csv_file in the annotationsfolder

    Arguments:
        filename (str): how you want to name the output file with the agreement scores
        kappa_dict (dict): dictionary with kappa score for each column
    """
    with open("../annotations/"+filename, 'w') as csvfile:
        writer = csv.writer(csvfile)

        for column, kappa in kappa_dict.items():
            writer.writerow([column, kappa])


if __name__ == "__main__":

    # takes filenames as commandline arguments
    filename_1 = sys.argv[1]
    filename_2 = sys.argv[2]

    # compute kappas
    kappa_dict = compute_agreement(filename_1, filename_2, threshold_score=0.8)

    # show for which categories there is not enough agreement
    for column, kappa in kappa_dict.items():
        if kappa >= 0.8:
            # sufficient agreement
            print("For the category", column,
                  "the kappa is:", kappa, u'\u2713')
        else:
            # insufficient agreement
            print("For the category", column,
                  "the kappa is:", kappa, u'\u274C')

    # save results in csv file
    out_filename = sys.argv[3]
    write_to_file(out_filename, kappa_dict)
