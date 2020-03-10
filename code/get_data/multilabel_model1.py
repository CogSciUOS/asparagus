from __future__ import print_function

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.utils import class_weight
#import tensorflow as tf
#import cv2

import keras
from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dense

from keras.utils import plot_model
from keras.utils import np_utils

from keras.callbacks import EarlyStopping

from grid import*
from submit_model import*


if __name__ == '__main__':
    ################################################################################
    # Set the variables
    ################################################################################
    args = typecast(sys.argv[1:])
    path_to_data = args[0]
    path_to_labels = args[1]

    ################################################################################
    # Load the data
    ################################################################################
    label_files = pd.read_csv(path_to_labels, sep=";")
    RELEVANT_COLUMNS = ['is_hollow', 'has_blume', 'has_rost_head', 'has_rost_body', 'is_bended', 'is_violet']
    labels = label_files[RELEVANT_COLUMNS].fillna(value = int(2))
    labels = labels.astype('int32')
    labels_train = labels.iloc[:12000]
    labels_test = labels.iloc[12000:]
    # hopefully this will create a column 'label' with all the other columns in a list
    labels_train['label'] = labels_train.values.tolist()
    print(labels_train.head())

    # desired datatype is a list with arrays containing the 6 labels seperated by a comma
    temp1 = (np.array(labels_train['label']))
    train_lbl = []
    for i in range(temp1.shape[0]):
        temp2 = str(temp1[i])
        temp3 = np.fromstring(temp2[1:-1], dtype = int, sep=',')
        train_lbl.append(temp3)
    
    train_lbl = np.array(train_lbl)
    # temp1 = (np.array(labels_train['label']))
    # for i in range(len(temp1)):
    #     temp2 = temp1[i]
    
    # temp2_lbl = temp1_lbl[:, np.newaxis]
    # train_lbl = [np.fromstring(temp2_lbl[i, 1:-1], dtype=int, sep=',') for i in range(len(temp1_lbl))]
    print(" >>> train_lbl.shape = ", train_lbl.shape)
    print(" >>> train_lbl at one pos = ", train_lbl[0])

    imgs = np.load(path_to_data)
    train_img = imgs[:12000]
    test_img = imgs[12000:]
    print(" >>> train_img.shape = ", train_img.shape)

    # muss ich auch noch normalisieren?
    # train_img = train_img.astype('float32') 
    # train_img /= 255

    ################################################################################
    # Build the model
    ################################################################################
    input_shape_img = (train_img.shape[1], train_img.shape[2], train_img.shape[3])
    batch_size = 32
    num_epochs = 50
    num_classes = 6
    conv_size = 32

    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape_img)) 
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same')) 
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same')) 
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(GlobalAveragePooling2D())
    model.add(Dense(num_classes, activation='sigmoid'))

    # add a costumize loss function that weights wrong labels for 1 higher than for 0 (because of class imbalance)
    weights = class_weight.compute_sample_weight(class_weight = "balanced", y = train_lbl)
    print(weights)
    def weighted_loss(y_true, y_pred):
        return K.mean((weights[:,0]**(1-y_true))*(weights[:,1]**(y_true))*K.binary_crossentropy(y_true, y_pred), axis=-1)
    
    
    model.compile(loss=weighted_loss, #'binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
                
    model.summary()

    ################################################################################
    # Train the model
    ################################################################################
    #early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)                               
    # class_weights = dict(zip(np.unique(train_lbl), class_weight.compute_sample_weight('balanced',
    #                                              np.unique(train_lbl),
    #                                              train_lbl)))
    
    history = model.fit(train_img, train_lbl,
                            batch_size=batch_size,
                            epochs=num_epochs,
                            verbose=1,
                            class_weight=class_weights, #{0:5, 1:3, 2:2 ,3:2 ,4:1 ,5:3},
                            validation_split=0.1)
                            #callbacks=[early_stop])

    ################################################################################
    # Check the history
    ################################################################################
    plt.figure(facecolor='white')

    # accuracy ---------------------------------------------------------------------
    ax1 = plt.subplot(2,1,1)

    plt.plot([x * 100 for x in history.history['acc']], label="acc", color="blue")
    plt.plot([x * 100 for x in history.history['val_acc']], label="val_acc", color="red")

    plt.title('Accuracy History')
    plt.ylabel('accuracy')
    # plt.xlabel('epoch')

    plt.legend(['train', 'valid'], loc='lower right')

    plt.ylim(0, 1)
    plt.xticks(np.arange(0, num_epochs + 1, 5))
    plt.yticks(np.arange(0, 100.1, 10))
    ax1.yaxis.set_major_formatter(plt.FuncFormatter('{:.0f}%'.format))
    plt.grid()

    # loss -------------------------------------------------------------------------
    plt.subplot(2,1,2)

    plt.plot(history.history['loss'], label="loss", color="blue")
    plt.plot(history.history['val_loss'], label="val_loss", color="red")

    plt.title('Loss History')
    plt.ylabel('loss')
    plt.xlabel('epoch')

    plt.legend(['train', 'valid'], loc='lower left')

    plt.ylim(0)
    plt.xticks(np.arange(0, num_epochs + 1, 5))
    plt.grid()
    plt.show()    
    plt.savefig('/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/asparagus/code/get_data/fig_weighted.png')