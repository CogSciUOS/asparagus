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

    # remove unclassified rows and the column itself
    unclassifiable = df[df['unclassified'] == 1].index
    df.drop(unclassifiable, inplace=True)
    df.drop(columns=['unclassified'], inplace=True)
    log.info(df.head())

    # drop rows with NaN values
    df.dropna(inplace=True)

    def relative_path(path):
        """ convert path stated in label csv file to corresponding path of image dir folder
            if erroneous, put NaN and delete"""
        log.info(path)
        # to prevent false paths
        if path is None:
            log.info("Missing path")
            return None

        ready_path = Path(imagedir) / \
            Path(path).relative_to(Path(path).parents[2])

        if ready_path.is_file():
            return ready_path
        else:
            log.warning("Invalid path: %s", ready_path)
            return None

    # process the filenames for the three images
    splitted_into_3_series = df['filenames'].str.split(', ', expand=True)

    # make a separate column for each of the three pictures for one asparagus piece (named: image_a, image_b, image_c)
    for i, col in enumerate('abc'):
        # apply the relative path function to each entry
        df[f'image_{col}'] = splitted_into_3_series[i].transform(relative_path)

    # drop original filenames column and NaN rows
    df = df.drop(columns='filenames')

    # drop rows with NaN values
    df.dropna(inplace=True)

    return df


def load_image(inputs, targets):
    """loads images based on path and adds them to the dataset

    Args:
        inputs ([type]): [description]
        targets ([type]): [description]

    Returns:
        tf dataset with images
    """
    img = tf.io.read_file(file_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    return inputs, targets


def create_dataset(df):
    """creates a tf dataset from the pandas df
    - defines input and target columns

    Args:
        df (pd dataframe): dataframe with features and and path to images

    Returns:
        tf dataset: tensorflow dataset with features and images of all available asparagus pieces
    """
    # these columns are generated
    auto_cols = [
        'auto_width',
        'auto_bended',
        'auto_length',
        'auto_violet',
    ]

    # these are the images
    image_cols = [
        'image_a',
        'image_b',
        'image_c'
    ]

    # these are the columns we want to predict
    # they were hand labeled by humans
    target_cols = [
        'is_hollow',
        'has_blume',
        'has_rost_head',
        'has_rost_body',
        'is_bended',
        'is_violet'
    ]

    # make strings out of the path objects
    for img_col in image_cols:
        df[img_col] = df[img_col].apply(str)

    # define inputs
    inputs = {
        "auto": df[auto_cols].values,
        "images": df[image_cols].values
    }
    # define outputs
    outputs = df[target_cols].values

    # https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_tensor_slices
    dataset = tf.data.Dataset.from_tensor_slices((inputs, outputs))
    # This transformation applies the function load_image to each element of this dataset,
    # and returns a new dataset containing the transformed elements
    dataset = dataset.map(load_image)

    return dataset


def main(labels, imagedir):
    # get preprocessed pd dataframe from csv file
    df = load_df(labels, imagedir)

    # create tensorflow dataset including images
    dataset = create_dataset(df)

    # look at the 5 first entries
    for feat, targ in dataset.take(1):
        print('Features: {}, Target: {}'.format(feat, targ))


if __name__ == "__main__":
    args = parse_args()
    main(args.labels, args.imagedir)
