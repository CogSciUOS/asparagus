import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow.keras as keras


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('labels', help='the combined labels csv')
    parser.add_argument('imagedir', help='the base image directory')
    return parser.parse_args()


def load_df(labels_csv, imagedir):
    """ loads the combined labels csv file as pandas df
        adds the correct path to the a, b, c images of each asparagus piece

    Args:
        labels_csv (string): path to the combined_labels_csv file
        imagedir (string): path to the image directory

    Returns:
        dataframe : df (each row is a feature vector of asparagus piece including path to images)
    """

    # columns that are going to be used
    usecols = [
        'is_hollow',
        'has_blume',
        'has_rost_head',
        'has_rost_body',
        'is_bended',
        'is_violet',
        'unclassified',
        'auto_width',
        'auto_bended',
        'auto_length',
        'auto_violet',
        'filenames',
    ]

    # convert to float
    def map_label(x):
        try:
            return float(x)
        except ValueError:
            return float('nan')

    # prepare smart read in with pandas
    # give column name and function that will be used for conversion on this column
    converters = {
        # remove brackets and quotes
        'filenames': lambda value: value[1:-1].replace("'", ""),
        'is_hollow': map_label,
        'has_blume': map_label,
        'has_rost_head': map_label,
        'is_bended': map_label,
        'is_violet': map_label,
    }

    # read in csv file and drop unnecessary columns and convert numbers to floats
    df = pd.read_csv(labels_csv, sep=';', usecols=usecols,
                     converters=converters)

    # remove unclassified rows
    unclassifiable = df[df['unclassified'] == 1].index
    df.drop(unclassifiable, inplace=True)

    def relative_path(path):
        """ convert path stated in df to corresponding path of image dir folder"""
        return Path(imagedir) / Path(path).relative_to(Path(path).parents[2])

    # process the filenames for the images
    split = df['filenames'].str.split(', ', expand=True)
    # make a separate column for each of the three pictures for one asparagus piece (named: image_a, image_b, image_c)
    for i, col in enumerate('abc'):
        df[f'image_{col}'] = split[i].transform(relative_path)

    # drop original filenames column and NaN rows
    df = df.drop(columns='filenames')
    df = df.dropna()

    return df


def main(labels, imagedir):
    df = load_df(labels, imagedir)
    df.head()


if __name__ == "__main__":
    args = parse_args()
    main(args.labels, args.imagedir)
