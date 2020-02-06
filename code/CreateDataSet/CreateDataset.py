
# in jupyter notebook oder so
# !pip install -q tensorflow tensorflow-datasets matplotlib

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import cv2

from tensorflow.keras.preprocessing.image import ImageDataGenerator


image_folder = ''

def read_in_data(path):
    '''
    '''
    images = []
    for filename in os.listdir(path):
        if filename[0] == '.':
            continue
        img = cv2.imread(os.path.join(path, filename))
        if img is not None:
            images.append(img)
    
    return np.array(images)


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