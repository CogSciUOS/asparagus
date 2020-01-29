""" This module computes the kappa agreement scores of our labelling outputs.
   Please provide 3 commandline arguments:
   1) the first annotator csv file
   2) the second annotator csv file
   3) the outputfile name ("agreement" + first_name + second_name)

   Example: python kappa_agreement.py ../annotations/annotator_1.csv ../annotations/annotator_2.csv agreement_annotator_1_annotator_2.csv
"""
import argparse
import csv
import itertools
import sys
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score


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

    # if unclassifiable, fill with value 2
    unclassified1 = annotations_1["unclassified"] == 1
    annotations_1[unclassified1] = annotations_1[unclassified1].fillna(2)

    unclassified2 = annotations_2["unclassified"] == 1
    annotations_2[unclassified2] = annotations_2[unclassified2].fillna(2)

    if drop_columns_starting_with is None:
        drop_columns_starting_with = ['auto', 'is_bruch', 'very_thick',
                                      'thick', 'medium_thick', 'thin', 'very_thin', 'unclassified', 'filenames']

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
        a dictionary of {category: f1score}
    """
    if np.any(annotations_1.isna()) or np.any(annotations_2.isna()):
        raise ValueError(
            "There are missing values in the csv file. Fix it and come back here!")
    return {column: f1_score(annotations_1[column], annotations_2[column], average='weighted') for column in annotations_1}


def write_to_file(filename, kappa_dict, accuracy_dict, f1_dict):
    """takes a dictionary of kappas and writes them into a csv_file in the annotationsfolder

    Arguments:
        filename (str): how you want to name the output file with the agreement scores
        kappa_dict (dict): dictionary with kappa score for each column
    """
    with open("../annotations/"+filename, 'w') as csvfile:
        writer = csv.writer(csvfile)

        # write header to file
        header = ["feature", "evaluation_measure", "score", "annotators"]
        writer.writerow(header)

        # TODO
        # I have to adjust this
        annotator_names = filename[filename.index("_")+22:-18]

        # write dict values to file
        for column, kappa in kappa_dict.items():
            writer.writerow(
                [column, "kappa score", kappa, str(annotator_names)])

        for column, acc in accuracy_dict.items():
            writer.writerow([column, "accuracy score",
                             acc, str(annotator_names)])

        for column, f1 in f1_dict.items():
            writer.writerow([column, "f1 score", f1, str(annotator_names)])


def main(args):
    annotations_1, annotations_2 = load_annotations(
        args.infile_1, args.infile_2)

    # Compute kappas
    print("Kappas:")
    print("Scores above .8 are generally considered good agreement; zero or lower means no agreement (practically random labels)!")
    kappa_dict = compute_agreement(annotations_1, annotations_2)
    for column, kappa in kappa_dict.items():
        rating = '\u2713' if kappa >= 0.8 else '\u274C'
        print(f"For the category {column}, the kappa is: {kappa} {rating}")
    print()

    # Compute accuracy
    print("Accuracy:")
    accuracy_dict = compute_accuracy(annotations_1, annotations_2)
    for column, accuracy in accuracy_dict.items():
        rating = '\u2713' if accuracy >= 0.8 else '\u274C'
        print(
            f"For the category {column}, the accuracy is: {accuracy} {rating}")
    print()

    # Compute f1 score
    print("F1 scores:")
    f1_dict = compute_f1_score(annotations_1, annotations_2)
    for column, f1 in accuracy_dict.items():
        rating = '\u2713' if f1 >= 0.8 else '\u274C'
        print(f"For the category {column}, the f1 score is: {f1} {rating}")
    print()

    # Write to file
    write_to_file(args.outfile, kappa_dict, accuracy_dict, f1_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('infile_1', help='the first annotator csv file')
    parser.add_argument('infile_2', help='the second annotator csv file')
    parser.add_argument('outfile', help='the outputfile name')

    #args = parser.parse_args()
    # main(args)

    # if you want to have all combinations of files in a folder
    path = Path("../annotations/evaluation_agreement_2")
    files = list(f for f in path.iterdir() if not f.name.startswith('agree'))
    for if1, if2 in itertools.combinations(files, 2):
        print(if1, if2)
        annotator1 = if1.name[:if1.name.index("_")]
        annotator2 = if2.name[:if2.name.index("_")]
        image_range = if1.name[if1.name.index("_")+1:]
        outname = path / \
            f"agreement_{annotator1}_{annotator2}_{image_range}"
        print(outname)

        args = parser.parse_args([str(if1), str(if2), str(outname)])
        main(args)
