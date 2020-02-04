
# in jupyter notebook oder so
# !pip install -q tensorflow tensorflow-datasets matplotlib

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


from tf.keras.preprocessing.image import ImageDataGenerator
# import tensorflow_datasets as tfds

# print(tf.__version__)

image_folder = ''



# training data configuration. Images will be sheared, zoomed, rescaled, flipped etc to 
# have a broader variety and hence better training
# this is helpful for conquering OVERfitting; when encounterin UNDERfitting, reduce preprocessing
train_generator = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

# testing generator. To have more variety of images that the network should
# be tested with. In this case, the images will only be rescaled, but could
# in theory be more altered  
test_generator = ImageDataGenerator(rescale=1. / 255)


# read in images and corresponding labels from directory

train_generator.flow_from_dictionary()





########################################################################

# https://www.tensorflow.org/guide/eager
# tf.compat.v1.enable_eager_execution() 


# List the available datasets

# Each dataset is implemented as a tfds.core.DatasetBuilder 
# and you can list all available builders with tfds.list_builders().
#tfds.list_builders()


# tfds.load: A dataset in one line
# tfds.load is a convenience method that's the simplest way to build and load a tf.data.Dataset.
# tf.data.Dataset is the standard TensorFlow API to build input pipelines. 
# Below, we load the MNIST training data. It downloads and prepares the data, unless you specify download=False.
# Note that once data has been prepared, subsequent calls of load will reuse the prepared data.
# You can customize where the data is saved/loaded by specifying data_dir= ( defaults to ~/tensorflow_datasets/).

# mnist_train = tfds.load(name="mnist", split="train")
# assert isinstance(mnist_train, tf.data.Dataset)

# print returns
# <_OptionsDataset shapes: {image: (28, 28, 1), label: ()}, types: {image: tf.uint8, label: tf.int64}> 
#print(mnist_train)



# When loading a dataset, the canonical default version is used. 
# It is however recommended to specify the major version of the dataset to use, 
# and to advertise which version of the dataset was used in your results. 

# OUTPUT IS A DICT
# mnist = tfds.load("mnist:1.*.*")

# accesses all entrys of the loaded dataset (aka dict)
# for MNIST, this returns 'train' and 'test'
# for element in mnist:
#    print(element)


# on the other hand, mnist.items()  returns:
# test <_OptionsDataset shapes: {image: (28, 28, 1), label: ()}, types: {image: tf.uint8, label: tf.int64}>
# train <_OptionsDataset shapes: {image: (28, 28, 1), label: ()}, types: {image: tf.uint8, label: tf.int64}>
# for key, value in mnist.items(): 
#     print (key, value)
