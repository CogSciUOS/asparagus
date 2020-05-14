# -*- coding: utf-8 -*-
#
# Author: Josefine Zerbe
"""
make_roc_curve.py
------------

Create a ROC curve for your model(s).

"""


## IMPORT LIBRARIES AND DEFINE PATHS ------------------------------------------

import datetime
import tensorflow as tf
import numpy as np
import os.path as op
import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras
from tensorflow.keras.layers import (Layer, Conv2D, MaxPool2D, Flatten, Dense,
                                     BatchNormalization)

print("Code started: ", datetime.datetime.now(), "\n")

# Define image path and label path.
project_dir = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus'
image_file = 'preprocessed_images/stacked_horizontal/0/data_horizontal.npy'
label_file = 'josefine/data/combined_new.csv'
model_folder = 'josefine/asparanet/models'

image_path = op.join(project_dir, image_file)
label_path = op.join(project_dir, label_file)
model_path = op.join(project_dir, model_folder)
             

## LOAD AND PREPARE THE DATA --------------------------------------------------

# The features:
#
# [0 = 'id',              1 = 'is_bruch',      2 = 'is_hollow',
#  3 = 'has_blume',       4 = 'has_rost_head', 5 = 'has_rost_body',
#  6 = 'is_bended',       7 = 'is_violet',     8 = 'very_thick',
#  9 = 'thick',          10 = 'medium_thick', 11 = 'thin',
# 12 = 'very_thin',      13 = 'unclassified', 14 = 'auto_violet',
# 15 = 'auto_blooming',  16 = 'auto_length',  17 = 'auto_rust_head',
# 18 = 'auto_rust_body', 19 = 'auto_width',   20 = 'auto_bended']
#


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


def normalize(arr):
  """Normalize an array to a range of [0; 1]."""
  return arr / (np.max(arr))


def balanced_sample_index(data_array, n_samples):
  """Balance out train data by drawing equal number of samples."""
  
  positive_samples = np.where(data_array == 1)[0][:n_samples]
  negative_samples = np.where(data_array == 0)[0][:n_samples]

  return np.concatenate([positive_samples, negative_samples], axis=0)



## BUILD THE ASPARANET MODEL --------------------------------------------------

class AsparaNet(tf.keras.Model):
  """ Small convolutional network for asparagus classification."""
  def __init__(self):
    super(AsparaNet, self).__init__()

    # Define the layers of the network.
    self.conv_1 = Conv2D(8, (5,  5), padding="valid", strides=(2, 2),
                         activation=None)
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


## FUNCTION DEFINITIONS -------------------------------------------------------

# Initialize or load the model.
model = AsparaNet()
#model.load_weights('models/04-17_16-05_is_bruch')

def get_confusion_matrices(y_pred,y_test,bias=False, additional_measures=False):
    """Calculate confusion matrix"""
    if bias == False:
        bias = .5
    y_pred = y_pred > bias
    y_test = pd.DataFrame(y_test, dtype=np.int32)
    y_pred = np.array(y_pred, dtype=np.int32)
    y_test = pd.DataFrame(y_test)
    #y_test is 1 while y_pred did not indicate
    false_negatives = np.sum(np.logical_and(y_test == 1,y_pred==0),axis=0)
    #y_test is 0 while y_pred say it was 1
    false_positives = np.sum(np.logical_and(y_test == 0,y_pred==1),axis=0)
    #both indicate 1
    true_positives = np.sum(np.logical_and(y_test == 1,y_pred==1),axis=0)
    #both indicate 0
    true_negatives = np.sum(np.logical_and(y_test == 0,y_pred==0),axis=0)
    summary = pd.DataFrame()
    summary['False positive'] = false_positives
    summary['False negative'] = false_negatives
    summary['True positive'] = true_positives
    summary['True negative'] = true_negatives
    summary = pd.DataFrame(summary)
    #print(summary.sum(axis=1)[0])
    summary_percent = (summary/summary.sum(axis=1)[0])
    if additional_measures:
        #summary_percent['Accuracy'] = summary_percent['True positive'] + summary_percent['True negative']
        summary_percent['Sensitivity'] = summary_percent['True positive']/(summary_percent['True positive']+summary_percent['False negative'])
        summary_percent['Specificity'] = summary_percent['True negative']/(summary_percent['True negative']+summary_percent['False positive'])

    return summary, summary_percent


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def roc_characteristic(y_pred,y_test):
    fpr = []
    tpr = []
    biases = []
    for bias in np.linspace(-50,50,1000):
        print(".",end="")
        summary, _ = get_confusion_matrices(y_pred,y_test,bias=np.abs(sigmoid(bias)))
        fpv = summary["False positive"]/(summary["False positive"]+summary["True negative"])
        tpv = summary["True positive"]/(summary["True positive"]+summary["False negative"])
        biases.append(bias)
        fpr.append(fpv)
        tpr.append(tpv)
    return np.array([np.array(fpr), np.array(tpr)])

fig, ax = plt.subplots(1,figsize=(10,7))


## LOOP FOR ROC CURVE ---------------------------------------------------------

# Save model names and features in order.
first_model_list = ["04-26_11-52_is_hollow",
                    "04-27_19-39_has_blume", "04-26_16-14_has_rost_head",
                    "04-26_14-49_has_rost_body", "04-28_22-47_is_bended",
                    "04-28_19-51_is_violet"]
second_model_list = ["04-17_21-17_is_bruch", "04-19_22-29_very_thick", "04-20_23-58_thick",
                     "04-21_01-26_medium_thick", "04-21_03-30_thin",
                     "04-20_00-28_very_thin", "04-20_12-55_unclassified"]

first_feature_list = [2, 3, 4, 5, 6, 7]
second_feature_list = [1, 8, 9, 10, 11, 12, 13]

featurename = ["fractured", "hollow", "flower", "rusty head",
               "rusty body","bent", "violet", "very thick",
               "thick", "medium thick", "thin", "very thin",
               "not classifiable"]

# Choose the feature list
model_list = first_model_list
feature_list = first_feature_list


current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")
filename = current_time + '_' + 'roc'
  
# Load the image data.
full_images = np.load(image_path)

# get the seeded shuffle index to recreate the test set
train_n, test_n = 12000, 1319
rnd_seed = 13477

for feature, model_name in zip(feature_list, model_list):

  # Initialize model
  model = AsparaNet()
  model.load_weights(op.join(model_path, model_name))

  # Call the function read_labels().
  indices, labels, feature_indicator, not_classifiable = read_labels(label_path,
                                                                   feature)

  batch_images = full_images

  # Remove unclassifiables for all features except 'is_bruch' & 'unclassified'.
  if feature not in (1, 13):
    for index in reversed(not_classifiable):
      batch_images = np.delete(batch_images, index, 0)
      labels = np.delete(labels, index, 0)

  # recreate the shuffle index
  shuffle_index = np.random.RandomState(seed=rnd_seed).permutation(len(labels))
  
  # print to check the used indices
  print("Indices used: ", shuffle_index)
      

  # Take only every 10th image
  #indices = balanced_sample_index(labels, 50)
  indices = shuffle_index[train_n:train_n + test_n]
  batch_images = batch_images[indices]
  labels = labels[indices]

  # Normalize the images.
  batch_images = normalize(batch_images)

  # Define and batch the test dataset.
  test_dataset = tf.data.Dataset.from_tensor_slices((batch_images, labels))
  test_dataset = test_dataset.batch(test_n)

  for (x,t) in test_dataset:
    y = model(x)
    
  roc = roc_characteristic(y, t)
  
  ax.plot(roc[0][:,0],roc[1][:,0], label=featurename[feature - 1])

  # Delete the full image array
  del batch_images

ax.plot([0,1],color = "gray")
ax.legend()

ax.set_ylabel('True positive rate')
ax.set_xlabel('False positive rate')
ax.set_title('Receiver operating characteristic (ROC)')
plt.savefig('roc/' + filename)
