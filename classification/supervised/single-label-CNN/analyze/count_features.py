# -*- coding: utf-8 -*-
#
# Author: Josefine Zerbe
"""
count_features.py
------------

Count the number of samples per feature in the hand-labelled asparagus data
set.

"""



## IMPORT LIBRARIES AND DEFINE PATHS ------------------------------------------

import datetime
import numpy as np
import os.path as op
import os

print("Code started: ", datetime.datetime.now(), "\n")


# Define image path and label path.
project_dir = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus'
image_file = 'preprocessed_images/stacked_horizontal/0/data_horizontal.npy'
label_file = 'josefine/data/combined_new.csv'

image_path = op.join(project_dir, image_file)
label_path = op.join(project_dir, label_file)
             


## LOAD THE DATA --------------------------------------------------------------


def read_labels(label_path, feature):
  """Create an array of the labels for the specific feature."""
  indices = []
  labels = []
  not_classifiable = []
  
  with open(label_path) as f:    
    for ind, line in enumerate(f):
      seg = line.split(";")

      # return the feature indicator
      if ind == 0:  
        feature_indicator = seg[feature]

      # skip the first line (which are the column titles)  
      if ind != 0:  
        indices.append(seg[0])

        # special case feature 1
        feature = 16 if feature == 1 else feature
        feature = 19 if feature in (8,9,10,11,12) else feature

        # special case feature 13
        if seg[13] in ("1.0", "1"):                      
          not_classifiable.append(ind - 1)
          
        # replace empty values  
        label = seg[feature] if seg[feature] != "" else "0.0"  
        labels.append((label))
        
  labels = np.array(labels).astype(np.float32)
  
  return indices, labels, feature_indicator, not_classifiable



## COUNT SAMPLES PER FEATURE --------------------------------------------------


# Go through every feature and count the occurences.
for feat in range(20):

  feature = feat

  indices, labels, feature_indicator, not_classifiable = read_labels(label_path, feature)
  
  if feature == 1:
    labels = (labels < 210).astype(np.float32)
  
  else:      #if feature not in (0, 13):
    if feature == 8:
      labels = (labels > 26).astype(np.float32)
    if feature == 9:
      labels = np.logical_and(labels > 20, labels <= 26).astype(np.float32)
    if feature == 10:
      labels = np.logical_and(labels > 18, labels <= 20).astype(np.float32)
    if feature == 11:
      labels = np.logical_and(labels > 16, labels <= 18).astype(np.float32)
    if feature == 12:
      labels = (labels <= 16).astype(np.float32)


  print("Feature = ", feature_indicator,
        "\n Ratio of positive samples = ", np.mean(labels),
        "\n Number of positive samples = ", int(np.sum(labels)),
        "\n Total samples = ", len(labels))

print("Code done: ", datetime.datetime.now())

