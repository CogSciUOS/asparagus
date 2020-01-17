
# in jupyter notebook oder so
# !pip install -q tensorflow tensorflow-datasets matplotlib


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds

# https://www.tensorflow.org/guide/eager
tf.compat.v1.enable_eager_execution() 


# List the available datasets

# Each dataset is implemented as a tfds.core.DatasetBuilder 
# and you can list all available builders with tfds.list_builders().
tfds.list_builders()


# tfds.load: A dataset in one line
# tfds.load is a convenience method that's the simplest way to build and load a tf.data.Dataset.
# tf.data.Dataset is the standard TensorFlow API to build input pipelines. 
# Below, we load the MNIST training data. It downloads and prepares the data, unless you specify download=False.
# Note that once data has been prepared, subsequent calls of load will reuse the prepared data.
# You can customize where the data is saved/loaded by specifying data_dir= ( defaults to ~/tensorflow_datasets/).

mnist_train = tfds.load(name="mnist", split="train")
assert isinstance(mnist_train, tf.data.Dataset)
print(mnist_train)

# When loading a dataset, the canonical default version is used. 
# It is however recommended to specify the major version of the dataset to use, 
# and to advertise which version of the dataset was used in your results. 

mnist = tfds.load("mnist:1.*.*")




