import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from logging import getLogger, StreamHandler, WARNING

import tensorflow.keras as keras
import tensorflow as tf

from tensorflow.keras.preprocessing.image import load_img

import matplotlib.pyplot as plt

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
            log.info("Invalid path: %s", ready_path)
            return None

    # process the filenames for the three images
    splitted_into_3_series = df['filenames'].str.split(', ', expand=True)

    # make a separate column for each of the three pictures for one asparagus piece (named: image_a, image_b, image_c)
    for i, col in enumerate('abc'):
        # apply the relative path function to each entry
        df[f'image_{col}'] = splitted_into_3_series[i].transform(
            relative_path)

    # drop original filenames column and NaN rows
    df = df.drop(columns='filenames')

    # drop rows with NaN values
    df.dropna(inplace=True)

    return df


def load_image(inputs, targets):
    """loads images based on path and adds them to the dataset
       see: https://www.tensorflow.org/tutorials/load_data/images

    Args:
        inputs  : input of dataset
        targets : target of dataset

    Returns:
        tf dataset with images
    """
    for img in 'abc':
        key = f'image_{img}_input'
        # load the raw data from the file as a string
        img = tf.io.read_file(inputs[key])
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_png(img, channels=3)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)
        # resize the image to the desired size.
        # img = tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])
        inputs[key] = img

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
        'auto_input': df[auto_cols].values,
        'image_a_input': df[image_cols[0]].values,
        'image_b_input': df[image_cols[1]].values,
        'image_c_input': df[image_cols[2]].values,
    }

    # define outputs
    outputs = df[target_cols].values

    # https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_tensor_slices
    dataset = tf.data.Dataset.from_tensor_slices((inputs, outputs))
    # This transformation applies the function load_image to each element of this dataset,
    # and returns a new dataset containing the transformed elements
    dataset = dataset.map(load_image)

    return dataset


def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # Repeat forever
    #ds = ds.repeat()
    ds = ds.repeat(3)

    ds = ds.batch(BATCH_SIZE)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


def show_batch(image_batch, target_batch):
    fig = plt.figure(figsize=(10, 10))
    for n in range(9):
        ax = plt.subplot(3, 3, n+1)
        plt.imshow(image_batch[n])
        plt.title(f"{target_batch[n]}")
        plt.axis('off')
    fig.savefig('plot.png')
    return fig


def get_compiled_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(4, ), name='auto_input'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(6)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])

    model.summary()

    return model


def plot_batch_sizes(ds):
    fig = plt.figure()
    # i changed this to stupid
    batch_sizes = [len(batch) for batch in ds]
    plt.bar(range(len(batch_sizes)), batch_sizes)
    plt.xlabel('Batch number')
    plt.ylabel('Batch size')
    fig.savefig('batch_size.png')


IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE

IMAGE_SHAPE = (1340, 364, 3)


def main(labels, imagedir):
    # get preprocessed pd dataframe from csv file
    df = load_df(labels, imagedir)

    # create tensorflow dataset including images
    dataset = create_dataset(df).batch(10)

    # look at the first entries
    for feat, targ in dataset.take(1):
        print("Feature shape: ", feat)
        print("Target shape: ", targ.numpy())
    print()
    print()

    print(dataset)

    # copied part from tutorial
    # train_ds = prepare_for_training(dataset)
    # image_batch, label_batch = next(iter(train_ds))
    # fig = show_batch(image_batch.numpy(), label_batch.numpy())
    # plt.show()

    # train_dataset = train_ds.shuffle(len(df)).batch(1)
    # plot_batch_sizes(train_dataset)

    model = get_compiled_model()

    model.fit(dataset, epochs=100)


if __name__ == "__main__":
    args = parse_args()
    main(args.labels, args.imagedir)
