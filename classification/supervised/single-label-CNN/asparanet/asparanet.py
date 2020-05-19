# -*- coding: utf-8 -*-
#
# Author: Josefine Zerbe
"""
asparanet.py
------------

Train a convolutional neural network to classify single features in images of
asparagus spears.  It is a single-label approach, in that you can choose one
feature from a set of features and the network returns whether the feature is
present in the image or not.


The script is divided into three sections.

In the first section, the data is loaded and processed.  The feature is
selected, then the data is shuffled and divided into a train set and test set.
Further, the batchsize is defined, the images are normalized and data
augmentation is applied.

In the second section, the model is defined. The AsparaNet model is a much
smaller version of AlexNet, with two convolutional layers separated by a
pooling layer, a hidden dense layer and an output layer.

In the third section, the model is trained.  The model is called, learning rate
and epochs are defined. In the training loop the model trains on the data and
every 10 steps it is tested on the validation data.

"""


## IMPORT LIBRARIES AND DEFINE PATHS ------------------------------------------

import datetime
import tensorflow as tf
import numpy as np
import os.path as op
import os
import pickle
from tensorflow import keras
from tensorflow.keras.layers import (Layer, Conv2D, MaxPool2D, Flatten, Dense,
                                     BatchNormalization)

print("Code started: ", datetime.datetime.now(), "\n")

# Define image path and label path.
project_dir = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus'
image_file = 'preprocessed_images/stacked_horizontal/0/data_horizontal.npy'
label_file = 'josefine/data/combined_new.csv'

image_path = op.join(project_dir, image_file)
label_path = op.join(project_dir, label_file)
             

## LOAD AND PREPARE THE DATA --------------------------------------------------

# Define the feature to be predicted:
#
# [0 = 'id',              1 = 'is_bruch',      2 = 'is_hollow',
#  3 = 'has_blume',       4 = 'has_rost_head', 5 = 'has_rost_body',
#  6 = 'is_bended',       7 = 'is_violet',     8 = 'very_thick',
#  9 = 'thick',          10 = 'medium_thick', 11 = 'thin',
# 12 = 'very_thin',      13 = 'unclassified', 14 = 'auto_violet',
# 15 = 'auto_blooming',  16 = 'auto_length',  17 = 'auto_rust_head',
# 18 = 'auto_rust_body', 19 = 'auto_width',   20 = 'auto_bended']
#
# Only feature 1 to 13 make sense for prediction, other features are not
# clearly labeled as 1.0 (= feature present) or 0.0 (= feature absent).
feature = 5

# Define the image number for the train set and the test set.
train_n = 12000
test_n = 1319     # train_n + test_n <= 13319

# Define the random seed.
rnd_seed = 13477     #92357038, 13477, 34213988
#np.random.seed(random_seed)
print("Random Seed: ", rnd_seed)

def read_labels(label_path, feature):
  """Create an array of the labels for the specific feature."""
  indices = []
  labels = []
  not_classifiable = []
  
  with open(label_path) as f:    
    for ind, line in enumerate(f):
      seg = line.split(";")

      # Return the feature indicator.
      if ind == 0:  
        feature_indicator = seg[feature]

      # Skip the first line (which are the column titles).  
      if ind != 0:  
        indices.append(seg[0])

        # Define columns to read depending on the feature.
        if feature == 1:
            read_col = 16
            
        elif feature in (8, 9, 10, 11, 12):
            read_col = 19
            
        else:
            read_col = feature

        # Special case for not classifiable samples.
        if seg[13] in ("1.0", "1"):                      
          not_classifiable.append(ind - 1)
          
        # Replace empty values with 0.0.  
        label = seg[read_col] if seg[read_col] != "" else "0.0"  
        labels.append((label))
        
  labels = np.array(labels).astype(np.float32)

  # Create logical indices for features relying on length or width.
  if feature == 1:  #  is_broken
      labels = (labels < 210).astype(np.float32)
      
  elif feature == 8: # very_thick
      labels = (labels > 26).astype(np.float32)

  elif feature == 9: # thick
      labels = np.logical_and(labels <= 26, labels > 20).astype(np.float32)

  elif feature == 10: # medium_thick
      labels = np.logical_and(labels <= 20, labels > 18).astype(np.float32)

  elif feature == 11: # thin
      labels = np.logical_and(labels <= 18, labels >= 16).astype(np.float32)
        
  elif feature == 12: # very_thick
      labels = (labels < 16).astype(np.float32) 

  return indices, labels, feature_indicator, not_classifiable

# Call the function read_labels().
indices, labels, feature_indicator, not_classifiable = read_labels(label_path,
                                                                   feature)

# Load the image data.
full_images = np.load(image_path)

# Remove unclassifiables for all features except 'is_bruch' and 'unclassified'.
if feature not in (1, 13):
    for index in reversed(not_classifiable):
        full_images = np.delete(full_images, index, 0)
        labels = np.delete(labels, index, 0)

# Shuffle the data.
shuffle_index = np.random.RandomState(seed=rnd_seed).permutation(len(labels))

#shuffle_index = np.random.permutation(range(len(labels)))
full_images = full_images[shuffle_index]
labels = labels[shuffle_index]

# Divide the images into train images and test images.
train_images = full_images[:train_n]
test_images = full_images[train_n:train_n + test_n]

print("test indices: ", shuffle_index[train_n:train_n + test_n])

train_labels = labels[:train_n]
test_labels = labels[train_n:train_n + test_n]

# Delete the full image array to save memory.
del full_images

# Some prints to check the feature distribution in the data.
print("Feature: ", feature, " ", feature_indicator)
print("Number of positive samples: ",
      sum(test_labels) + sum(train_labels))
print("Ratio of positive samples in train labels: ",
      sum(train_labels) / len(train_labels))
print("Ratio of positive samples in test labels: ",
      sum(test_labels) / len(test_labels), "\n")


def normalize(arr):
  """Normalize an array to a range of [0; 1]."""
  return arr / (np.max(arr))

# Normalize the images.
train_images = normalize(train_images)
test_images = normalize(test_images)


def balance_index(data_array):
  """Balance out train data by duplicating minor sample category."""
  
  positive_samples = np.where(data_array == 1)[0]
  negative_samples = np.where(data_array == 0)[0]

  if len(positive_samples) < len(negative_samples):
    duplicates = len(negative_samples) // len(positive_samples)
    rest = len(negative_samples) % len(positive_samples)
    
    positive_samples = np.concatenate([positive_samples
                                       for i in range(duplicates)], axis=0)
    positive_samples = np.concatenate([positive_samples,
                                       positive_samples[:rest]], axis=0)
  
  else:
    duplicates = len(positive_samples) // len(negative_samples)
    rest = len(positive_samples) % len(negative_samples)
    
    negative_samples = np.concatenate([negative_samples
                                       for i in range(duplicates)], axis=0)
    negative_samples = np.concatenate([negative_samples,
                                       negative_samples[:rest]], axis=0)

  return np.concatenate([positive_samples, negative_samples], axis=0)

# Balance out the train data.
balanced_ids = balance_index(train_labels)

# Define and batch the test dataset.
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
test_dataset = test_dataset.batch(test_n)

# For training data, we use data augmentation to enhance generalization.
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    horizontal_flip = True,
    fill_mode='constant', 
    cval=np.min(train_images))


## BUILD THE ASPARANET MODEL --------------------------------------------------

class AsparaNet(tf.keras.Model):
  """ Small convolutional network for asparagus classification."""
  def __init__(self):
    super(AsparaNet, self).__init__()

    # Define the layers of the network.
    self.conv_1 = Conv2D(8, (5,  5), padding="valid", strides=(2, 2),
                         activation=None, input_shape=train_images.shape[1:])
    self.bn_1 = BatchNormalization()
    self.pool_1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="valid")
    self.conv_2 = Conv2D(16, (3, 3), padding="valid", strides=(1, 1),
                         activation=None)
    self.bn_2 = BatchNormalization()
    self.flatten_1 = Flatten()
    self.dense_1 = Dense(32, activation="relu")
    self.dense_2 = Dense(1, activation=tf.nn.sigmoid)

  def call(self, x, is_training=False):
    """Call the model."""
    x = self.conv_1(x)
    x = tf.keras.activations.relu(self.bn_1(x, training=is_training))
    x = self.pool_1(x)
    x = self.conv_2(x)
    x = tf.keras.activations.relu(self.bn_2(x, training=is_training))
    x = self.flatten_1(x)
    x = self.dense_1(x)
    x = self.dense_2(x)
    return x


## PREPARE TRAINING THE MODEL -------------------------------------------------

# Initialize the summary writers to store data for TensorBoard.
current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")
model_id = current_time + '_' + feature_indicator
train_log_dir = 'logs/gradient_tape/' + model_id + '/train'
test_log_dir = 'logs/gradient_tape/' + model_id + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

# Clear backend.
tf.keras.backend.clear_session()

# Initialize or load the model.
model = AsparaNet()
#model.load_weights('models/04-17_16-05_is_bruch')

# Define parameters for training loop.
epochs = 120
batch_size = 512
learning_rate = 1e-5

# Define the loss function.
cross_ent = tf.keras.losses.BinaryCrossentropy()


def l2_loss(loss, model, l2_factor=0.01):
    """L2 regularize loss."""
    l2_loss = 0
    for w in model.trainable_variables:
        l2_loss += tf.nn.l2_loss(w)
    return loss + l2_factor * l2_loss

# Initialize optimizer: Adam with default parameters.
optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)


## TRAIN THE MODEL ------------------------------------------------------------

step = 0

for epoch in range(epochs):
    
  # Shuffle and batch the index array.
  last_idx = len(balanced_ids) % batch_size
  shuffled_ids = np.random.permutation(balanced_ids)[:-(last_idx)]
  batched_ids = shuffled_ids.reshape(-1, batch_size)

  for cur_ids in batched_ids:
      
    # Get the data from our current indices.
    x = tf.convert_to_tensor(train_images[cur_ids])
    t = tf.expand_dims(tf.convert_to_tensor(train_labels[cur_ids]), axis=-1)
    
    # If needed, apply data augmentation.
    x = train_datagen.apply_transform(x, dict(flip_horizontal=False))

    # Compute the output, loss and the gradients.
    with tf.GradientTape() as tape:
      output = model(x, is_training=True)
      loss = cross_ent(t, output)
      loss_l2 = l2_loss(loss, model)
      gradients = tape.gradient(loss_l2, model.trainable_variables)

      # Calculate the train accuracy.
      accuracy = np.mean([t == tf.math.round(output)]) * 100

    # Apply the gradients.
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Store train loss and train accuracy in Tensorboard.
    with train_summary_writer.as_default():
      tf.summary.scalar('train loss', loss, step=step)
      tf.summary.scalar('train accuracy', accuracy, step=step)
      
      for index, gradient in enumerate(gradients):
        tf.summary.histogram("{}gradients".format(index), gradient, step=step)

    # Check the generalization.
    if step % 10 == 0:
        
      for (x_test,t_test) in test_dataset:
        
        # Feed model with test data.
        output = model(x_test, is_training=False)

        # Compute loss and accuracy for the test data.
        loss = cross_ent(t_test, output)
        accuracy = np.mean([tf.expand_dims(t_test, axis=-1)
                            == tf.math.round(output)]) * 100
        print("Step: ", step)
        print("Validation accuracy: ", accuracy)

        # Round the output to make clear prediction.
        prediction = np.round(output).reshape(-1,)
        
        # Compute true positives (TP) and true negatives (TN).           
        true_positives = np.where(np.logical_and(prediction == 1,
                                                 t_test == prediction))[0]
        true_negatives = np.where(np.logical_and(prediction == 0,
                                                 t_test == prediction))[0]

        # Compute false positives (FP) and false negatives (FN).
        false_positives = np.where(np.logical_and(prediction == 1,
                                                  t_test != prediction))[0]
        false_negatives = np.where(np.logical_and(prediction == 0,
                                                  t_test != prediction))[0]

        # Print ratio of TP and ratio of TN.
        print("TP: ", len(true_positives), " / ",
              len(true_positives) + len(false_negatives))
        print("TN: ", len(true_negatives), " / ",
              len(true_negatives) + len(false_positives))
        
        # Compute and print the true positive rate (TPR)
        tpr = len(true_positives) / (len(true_positives)
                                     + len(false_negatives))
        print("true positive rate: ", tpr)
        
        # Compute and print the true negative rate (TNR).
        tnr = len(true_negatives) / (len(true_negatives)
                                     + len(false_positives))
        print("true negative rate: ", tnr)
        print("balanced accuracy: ", (tpr + tnr) / 2, "\n")

        # Store TPR, TNR, test loss, and test accuracy in Tensorboard.
        with test_summary_writer.as_default():
          tf.summary.text('True Positive Rate(TPR) vs True Negative Rate(TNR)',
                          "TPR: " + str(tpr) + "; TNR: " + str(tnr), step=step)
          tf.summary.scalar('test loss', loss, step=step)
          tf.summary.scalar('test accuracy', accuracy, step=step)

    step += 1


## STORE SOME LAST INFORMATION ------------------------------------------------

# Check out FP and FN example images of last training step in Tensorboard.
with test_summary_writer.as_default():
    max_out = len(false_negatives) if (len(false_negatives) < 5) else 5
    tf.summary.image("False negative example", x_test.numpy()[false_negatives],
                     max_outputs=max_out, step=step)
    max_out = len(false_positives) if (len(false_positives) < 5) else 5
    tf.summary.image("False positive example", x_test.numpy()[false_positives],
                     max_outputs=max_out, step=step)

# Save the model weights.
model.save_weights('models/' + model_id)
          
# Some last prints for overview.
print("Learning rate: ", learning_rate)
print("Performed steps: ", step)

epoch_progress = batch_size * step / len(balanced_ids)
print("Performed epochs: ", epoch_progress)
print("Code done: ", datetime.datetime.now())
