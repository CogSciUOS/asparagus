""" This module computes the kappa agreement scores, the accuracy and the f1 score of our labelling outputs.
   Please provide 3 commandline arguments:
   1) the first annotator csv file
   2) the second annotator csv file
   3) the outputfile name ("agreement" + first_name + second_name)

   Example: python kappa_agreement.py ../annotations/annotator_1.csv ../annotations/annotator_2.csv agreement_annotator_1_annotator_2.csv
"""
import argparse
import sys
import csv

import pandas as pd
import numpy as np

from sklearn.metrics import cohen_kappa_score, classification_report, accuracy_score, f1_score


def load_annotations(filename_1, filename_2, drop_columns_starting_with=None):
    """This functions loads two csv files of annotations.

    The first columns are used as index columns.

    Args:
        drop_columns_starting_with(list): Columns starting with these strings will be dropped.
                                          If it is `None`, it defaults to
                                          ['auto', 'is_bruch', 'very_thick', 'thick', 'medium_thick', 'thin', 'very_thin', 'unclassified'].

    Returns:
        The two dataframes.
    """
    annotations_1 = pd.read_csv(filename_1, delimiter=";", index_col=0)
    annotations_2 = pd.read_csv(filename_2, delimiter=";", index_col=0)

    if drop_columns_starting_with is None:
        drop_columns_starting_with = ['auto', 'is_bruch', 'very_thick',
                                      'thick', 'medium_thick', 'thin', 'very_thin', 'unclassified']

    for column in drop_columns_starting_with:
        mask = annotations_1.columns.str.startswith(column)
        annotations_1 = annotations_1.loc[:, ~mask]

        mask = annotations_2.columns.str.startswith(column)
        annotations_2 = annotations_2.loc[:, ~mask]

    return annotations_1, annotations_2


def compute_agreement(annotations_1, annotations_2):
    """This function takes two dataframes of annotations (with labels 1.0 and 0.0)
    and computes Cohen’s kappa, a score that expresses the level of agreement between two annotators on a classification problem.

    Arguments:
        annotations_1 (pd.DataFrame): first annotations
        annotations_2 (pd.DataFrame): second annotations

    Returns:
        a dictionary of {category: kappa_score}
    """
    if np.any(annotations_1.isna()) or np.any(annotations_2.isna()):
        raise ValueError(
            "There are missing values in the csv file. Fix it and come back here!")
    return {column: cohen_kappa_score(annotations_1[column], annotations_2[column]) for column in annotations_1}


def compute_accuracy(annotations_1, annotations_2):
    """ This function takes two dataframes of annotations (with labels 1.0 and 0.0)
    and computes the accuracy, another score that expresses the level of agreement
    between two annotators on a classification problem.

    Arguments:
        annotations_1 (pd.DataFrame): first annotations
        annotations_2 (pd.DataFrame): second annotations

    Returns:
        a dictionary of {category: accuracy}
    """
    if np.any(annotations_1.isna()) or np.any(annotations_2.isna()):
        raise ValueError(
            "There are missing values in the csv file. Fix it and come back here!")
    return {column: accuracy_score(annotations_1[column], annotations_2[column]) for column in annotations_1}


def compute_f1_score(annotations_1, annotations_2):
    """ This function takes two dataframes of annotations (with labels 1.0 and 0.0)
    and computes the accuracy, another score that expresses the level of agreement
    between two annotators on a classification problem.

    Arguments:
        annotations_1 (pd.DataFrame): first annotations
        annotations_2 (pd.DataFrame): second annotations

    Returns:
        a dictionary of {category: accuracy}
    """
    if np.any(annotations_1.isna()) or np.any(annotations_2.isna()):
        raise ValueError(
            "There are missing values in the csv file. Fix it and come back here!")
    return {column: f1_score(annotations_1[column], annotations_2[column], average='weighted') for column in annotations_1}


def compute_report(annotations_1, annotations_2):
    """Build a text report showing the main classification metrics.

    Arguments:
        annotations_1 (pd.DataFrame): first annotations
        annotations_2 (pd.DataFrame): second annotations

    Returns:
        a dictionary of {category: classification_report}
    """
    if np.any(annotations_1.isna()) or np.any(annotations_2.isna()):
        raise ValueError(
            "There are missing values in the csv file. Fix it and come back here!")
    return {column: classification_report(annotations_1[column], annotations_2[column]) for column in annotations_1}


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
    parser = argparse.ArgumentParser()
    parser.add_argument('infile_1', help='the first annotator csv file')
    parser.add_argument('infile_2', help='the second annotator csv file')
    parser.add_argument('outfile', help='the outputfile name')
    args = parser.parse_args()

    annotations_1, annotations_2 = load_annotations(
        args.infile_1, args.infile_2)

    # Compute kappas
    print("Scores above .8 are generally considered good agreement; zero or lower means no agreement (practically random labels)!\n")
    kappa_dict = compute_agreement(annotations_1, annotations_2)
    for column, kappa in kappa_dict.items():
        rating = '\u2713' if kappa >= 0.8 else '\u274C'
        print(f"For the category {column}, the kappa is: {kappa} {rating}")
    out_kappa = args.outfile + "_kappa.csv"
    write_to_file(out_kappa, kappa_dict)


    # compute accuracy
    accuracy_dict = compute_accuracy(annotations_1, annotations_2)
    for column, accuracy in accuracy_dict.items():
        print(f"For the category {column}, the accuracy is: {accuracy} {rating}")
    out_acc = args.outfile + "_accuracy.csv"a
    write_to_file(out_acc, accuracy_dict)

    # compute f1 score
    f1_dict = compute_f1_score(annotations_1, annotations_2)
    for column, f1 in accuracy_dict.items():
        print(f"For the category {column}, the f1 score is: {f1} {rating}")
    out_f1 = args.outfile + "_f1.csv"
    write_to_file(out_f1, f1_dict)

    # Compute report
    report_dict = compute_report(annotations_1, annotations_2)
    for column, classification in report_dict.items():
        print(f'{column:=^20}', '\n', classification)
