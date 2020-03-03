import argparse
from pathlib import Path
import os
import time

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


def get_compiled_model():
    """ define and compile the model"""
    auto_model = tf.keras.Sequential([
        # this is the auto input
        tf.keras.layers.Input(shape=(4, ), name='auto_input'),
        tf.keras.layers.Dense(1000, activation='relu'),
        tf.keras.layers.Dense(500, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(200, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dense(10, activation='sigmoid'),
        tf.keras.layers.Dense(6)
    ])

    image_model = tf.keras.Sequential([
        # this is the image input
        tf.keras.layers.Input(shape=IMAGE_SHAPE, name='image_a_input'),
        tf.keras.layers.Conv2D(filters=96, kernel_size=(
            11, 11), strides=(4, 4), padding='valid'),
        tf.keras.layers.MaxPooling2D(pool_size=(
            2, 2), strides=(2, 2), padding='valid'),
        tf.keras.layers.Conv2D(
            filters=96, kernel_size=(11, 11), padding='valid'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(
            filters=96, kernel_size=(11, 11), padding='valid'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(50),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(6)  # , activation='softmax'),
    ])

    auto_model.compile(optimizer='adam',
                       loss=tf.keras.losses.BinaryCrossentropy(),
                       metrics=['accuracy',
                                 'mse',
                                 # keras.metrics.TruePositives(),
                                 # keras.metrics.TrueNegatives(),
                                 # keras.metrics.FalsePositives(),
                                 # keras.metrics.FalseNegatives(),
                                ])

    auto_model.summary()

    return auto_model


def save_images(image_batch, target_batch):
    """ visualizes the asparagus pieces and shows target vector to be learned """
    # look at first batch
    batch = 0

    fig = plt.figure(figsize=(10, 10))
    for letter in "abc":
        plt.imshow(image_batch[f'image_{letter}_input'][batch][0])
        fig.savefig(f'test{letter}.png')

    # put 3 x 3 on the figure
    for n in range(9):
        ax = plt.subplot(3, 3, n+1)
        # TODO only seeing image_a !!!
        plt.imshow(image_batch['image_a_input'][batch][0])
        plt.title(
            f"batch: {n} \n targ: {target_batch[batch][0]} \n auto: {image_batch['auto_input'][batch][0]}")
        plt.axis('off')
    fig.tight_layout()
    fig.savefig('plot.png')

    return fig


# TODO use this script as a grid job
class EarlyStoppingAfterMinutes(keras.callbacks.Callback):
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


IMAGE_SHAPE = (1340, 364, 3)

# unused right now; can be used to resize the image
IMG_HEIGHT = 224
IMG_WIDTH = 224

# take 10 samples in parallel to update model
BATCH_SIZE = 10
AUTOTUNE = tf.data.experimental.AUTOTUNE


def main(labels, imagedir):
    # get preprocessed pd dataframe from csv file
    df = load_df(labels, imagedir)

    # create tensorflow dataset including images
    dataset = create_dataset(df).batch(5)
    print()
    print("The dataset:", dataset)
    print()

    # shuffle
    dataset = dataset.shuffle(buffer_size=10, seed=2)

    # split into validation und training
    val_dataset = dataset.take(100)
    train_dataset = dataset.skip(100)
    train_dataset = train_dataset.shuffle(3, reshuffle_each_iteration=True)

    # define and compile the model to be trained
    model = get_compiled_model()
    # fit the model to the data and validate
    model.fit(train_dataset, epochs=1, validation_data=val_dataset)

    # saving some images with target vectors
    image_batch, label_batch = next(iter(train_dataset))
    save_images(image_batch, label_batch.numpy())


if __name__ == "__main__":
    args = parse_args()
    main(args.labels, args.imagedir)
