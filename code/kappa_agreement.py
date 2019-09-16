from sklearn.metrics import cohen_kappa_score
import sys

import pandas as pd
import numpy as np


def compute_agreement(filename_1, filename_2, threshold_score=0.8):
    """This function takes two csv files of annotations (with labels 1.0 and 0.0)
    and computes Cohen’s kappa, a score that expresses the level of agreement between two annotators on a classification problem.

    Arguments:
        filename_1 {str} -- first annotation file name
        filename_2 {str} -- first annotation file name

    Keyword Arguments:
        threshold_score {float} -- determines whether you decide that agreement is acceptable or not (default: {0.8})
    """

    # read in files
    annotations_1 = pd.read_csv(filename_1, delimiter=";")
    annotations_2 = pd.read_csv(filename_2, delimiter=";")

    # Check for NaNs
    if np.any(annotations_1.isna()) or np.any(annotations_2.isna()):
        print("There are missing values in the csv file. Fix it and come back here!")
        sys.exit(1)

    print("!!! Scores above .8 are generally considered good agreement; zero or lower means no agreement(practically random labels) !!!")

    # calculate kappa for each category/column
    for column in annotations_1.columns[1:]:

        kappa = cohen_kappa_score(
            annotations_1[column], annotations_2[column], labels=[1.0, 0.0])

        if kappa >= 0.8:
            # sufficient agreement
            print("For the category", column,
                  "the kappa is:", kappa, u'\u2713')

        else:
            # insufficient agreement
            print("For the category", column,
                  "the kappa is:", kappa, u'\u274C')


if __name__ == "__main__":

    #filename_1 = "../annotations/annotator_1.csv"
    #filename_2 = "../annotations/annotator_2.csv"

    filename_1 = sys.argv[1]
    filename_2 = sys.argv[2]

    compute_agreement(filename_1, filename_2, threshold_score=0.8)
