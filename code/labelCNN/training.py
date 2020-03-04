import argparse
from pathlib import Path
import os
import time

import numpy as np
import pandas as pd
import h5py
from logging import getLogger, StreamHandler, WARNING

import tensorflow.keras as keras
import tensorflow as tf

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt


log = getLogger(__file__)
log.setLevel(WARNING)
log.addHandler(StreamHandler())

# Disable warnings
DISABLE_WARNINGS = True
if DISABLE_WARNINGS:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf  # noqa
    tf.get_logger().setLevel('ERROR')


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
        img = tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])
        inputs[key] = img

    return inputs, targets


def create_dataset(df, batch_size=5):
    """creates a tf dataset from the pandas df
    - defines input and target columns

    Args:
        df (pd dataframe): dataframe with features and and path to images
        batch_size (int): batch size

    Returns:
        train and validation dataset: tensorflow dataset with features and images of all available asparagus pieces
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

    # shuffle
    dataset = dataset.shuffle(buffer_size=10, seed=2)

    # split into validation und training and batch
    val_dataset = dataset.take(100).batch(batch_size)
    train_dataset = dataset.skip(100)
    train_dataset = train_dataset.shuffle(
        buffer_size=10, reshuffle_each_iteration=True).batch(batch_size)

    return train_dataset, val_dataset


def get_compiled_model():
    """ define and compile the model """

    image_a_in = keras.layers.Input(shape=IMAGE_SHAPE, name='image_a_input')
    image_b_in = keras.layers.Input(shape=IMAGE_SHAPE, name='image_b_input')
    image_c_in = keras.layers.Input(shape=IMAGE_SHAPE, name='image_c_input')

    concat_images = keras.layers.Concatenate(axis=-1)(
        [image_a_in, image_b_in, image_c_in])

    x = keras.layers.Conv2D(
        filters=40, kernel_size=(8, 8), padding='valid')(concat_images)
    x = keras.layers.Dropout(0.5)(x)
    # x = keras.layers.MaxPooling2D(pool_size=(
    #    2, 2), strides=(2, 2), padding='valid')(x)
    x = keras.layers.Conv2D(
        filters=30, kernel_size=(11, 11), padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(
        filters=20, kernel_size=(11, 11), padding='valid')(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Flatten()(x)
    out = keras.layers.Dense(6, activation='tanh',
                             name='output_images')(x)

    images_merged_model = keras.models.Model(
        [image_a_in, image_b_in, image_c_in], out)

    auto_in = keras.layers.Input(shape=(4, ), name='auto_input')
    y = keras.layers.Dense(64, activation="tanh")(auto_in)
    y = keras.layers.Dense(4, activation="tanh")(y)
    y_model = keras.models.Model(inputs=auto_in, outputs=y)

    combined = keras.layers.Concatenate()(
        [images_merged_model.output, y_model.output])

    z = keras.layers.Dense(10, activation="tanh")(combined)
    z = keras.layers.Dense(6, activation="tanh")(z)

    model = keras.models.Model(
        inputs=[images_merged_model.input, auto_in], outputs=z)

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy',
                           'mse',
                           # keras.metrics.TruePositives(),
                           # keras.metrics.TrueNegatives(),
                           # keras.metrics.FalsePositives(),
                           # keras.metrics.FalseNegatives(),
                           ])

    model.summary()

    return model


class EarlyStoppingAfterMinutes(keras.callbacks.Callback):
    """ to respect wall time of grid jobs"""

    def __init__(self, minutes):
        self.timeout = minutes * 60
        self.start = None
        self.last_epoch_start = 0
        self.epochs = 0
        self.mean_time = 0

    def on_train_begin(self, logs=None):
        self.start = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self.last_epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        now = time.time()
        approx_total = self.mean_time * \
            self.epochs + (now - self.last_epoch_start)
        self.epochs += 1
        self.mean_time = approx_total / self.epochs
        if now - self.start + 1.5 * self.mean_time > self.timeout:
            print('Stopping training, time is up')
            self.model.stop_training = True


# IMAGE_SHAPE = (1340, 364, 3)
IMAGE_SHAPE = (670, 182, 3)

# unused right now; can be used to resize the image
IMG_HEIGHT = 182
IMG_WIDTH = 670


def main(labels, imagedir):

    # get preprocessed pd dataframe from csv file
    df = load_df(labels, imagedir)[:150]

    # create tensorflow dataset including images separated in train and val set
    train_dataset, val_dataset = create_dataset(
        df, batch_size=5)

    # define and compile the model to be trained
    model = get_compiled_model()

    # fit the model to the data and validate
    callbacks = []
    model.fit(train_dataset, epochs=1,
              validation_data=val_dataset, callbacks=callbacks)

    #########################

    # try to predict
    print("These are the predictions for the first one of first batch of the validation set")
    sample = val_dataset.take(1)
    for feat, tar in sample:
        print(tar.numpy())
    batch = model.predict(sample)
    for row in batch:
        print(
            f"The rounded target prediction: {[int(round(val)) for val in row]}")
    print()
    print()
    for row in batch:
        print(
            f"The rounded target prediction: {[round(val, 1) for val in row]}")

    #################save and load again######################
    # creates a HDF5 file 'my_model.h5'
    model.save('my_model.h5')
    # deletes the existing model
    del model

    # returns a compiled model
    # identical to the previous one
    model = load_model('my_model.h5')

    prediction = model.predict(sample)[0]
    print(prediction)
    print(
        f"The rounded target prediction: {[round(val, 1) for val in prediction]}")


if __name__ == "__main__":
    args = parse_args()
    main(args.labels, args.imagedir)
