import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from logging import getLogger, StreamHandler, WARNING

import tensorflow.keras as keras
import tensorflow as tf


log = getLogger(__file__)
log.setLevel(WARNING)
log.addHandler(StreamHandler())


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
    df.drop(columns=['unclassified'], inplace=True)
    log.info(df.head())

    # drop rows with NaN values
    df.dropna(inplace=True)

    def relative_path(path):
        """ convert path stated in df to corresponding path of image dir folder
            if erroneous, put NaN and delete"""
        log.info(path)
        # to prevent false paths
        if path is None:
            log.warning("Missing path")
            return None

        ready_path = Path(imagedir) / \
            Path(path).relative_to(Path(path).parents[2])

        if ready_path.is_file():
            return str(ready_path)
        else:
            log.warning("Invalid path: %s", ready_path)
            return None

    # process the filenames for the images
    splitted_into_3_series = df['filenames'].str.split(', ', expand=True)

    # make a separate column for each of the three pichttps://pandas.pydata.org/pandas-docs/stable/user_guide/text.htmltures for one asparagus piece (named: image_a, image_b, image_c)
    for i, col in enumerate('abc'):
        df[f'image_{col}'] = splitted_into_3_series[i].transform(relative_path)

    # drop original filenames column and NaN rows
    df = df.drop(columns='filenames')

    # drop rows with NaN values
    df.dropna(inplace=True)

    input_cols = [
        'auto_width',
        'auto_bended',
        'auto_length',
        'auto_violet',
    ]

    images_cols = [
        'image_a',
        'image_b',
        'image_c'
    ]

    target_cols = [
        'is_hollow',
        'has_blume',
        'has_rost_head',
        'has_rost_body',
        'is_bended',
        'is_violet'
    ]

    for col in input_cols + target_cols:
        df[col] = df[col].astype("float32")

    for img_col in images_cols:
        df[img_col] = df[img_col].astype("string")

    return df[input_cols + images_cols], df[target_cols]


def main(labels, imagedir):
    input_df, target_df = load_df(labels, imagedir)
    print(input_df.head())
    print(target_df.head())

    print(type(input_df))
    print(type(target_df))

    print(input_df.dtypes)
    print(target_df.dtypes)

    dataset = tf.data.Dataset.from_tensor_slices(
        (input_df.values, target_df.values))

    for feat, targ in dataset.take(5):
        print('Features: {}, Target: {}'.format(feat, targ))


if __name__ == "__main__":
    args = parse_args()
    main(args.labels, args.imagedir)
