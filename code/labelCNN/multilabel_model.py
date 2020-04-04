from __future__ import print_function

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import hamming_loss
#from sklearn.utils import class_weight
#import tensorflow as tf
#import cv2

import keras.backend as K
from keras.losses import binary_crossentropy
from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout
from keras.layers import Dense

from keras.regularizers import l1, l2

#from keras.utils import plot_model
#from keras.utils import np_utils

#from keras.callbacks import EarlyStopping

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

    ################################################################################
    # Build the model
    ################################################################################
    input_shape_img = (train_img.shape[1], train_img.shape[2], train_img.shape[3])
    batch_size = 32
    num_epochs = 25
    num_classes = 6
    reg = l1(0.001)

    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape_img, kernel_regularizer=l2(0.01)))
    #model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(GlobalAveragePooling2D())
    model.add(Dense(num_classes, activation='sigmoid'))

    # add a costumize loss function that weights wrong labels for 1 higher than for 0 (because of class imbalance)
    def weighted_loss(y_true, y_pred):
        return K.mean((0.8**(1-y_true))*(1**(y_true))*K.binary_crossentropy(y_true, y_pred), axis=-1)
    
    def hamming(y_true, y_pred):
        return hamming_loss(y_true, y_pred)

    def hn_multilabel_loss(y_true, y_pred):
        # code snippet from https://groups.google.com/forum/#!topic/keras-users/_sjndHbejTY
        # Avoid divide by 0
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # Multi-task loss
        return K.mean(K.sum(- y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred), axis=1))
    
    def FN_wrapper():
        def falseNegatives(y_true, y_pred):
            neg_y_pred = 1 - y_pred
            fn = K.sum(y_true * neg_y_pred)
            return fn
        return falseNegatives

    def FP_wrapper():
        def falsePositives(y_true, y_pred):
            neg_y_true = 1 - y_true
            fp = K.sum(neg_y_true * y_pred)
            return fp
        return falsePositives

    def TN_wrapper():
        def trueNegatives(y_true, y_pred):
            neg_y_true = 1 - y_true
            neg_y_pred = 1 - y_pred
            tn = K.sum(neg_y_true * neg_y_pred)
            return tn
        return trueNegatives

    def TP_wrapper():
        def truePositives(y_true, y_pred):
            tp = K.sum(y_true * y_pred)
            return tp
        return truePositives

    FN = FN_wrapper()
    FP = FP_wrapper()
    TN = TN_wrapper()
    TP = TP_wrapper()

    model.compile(#loss=weighted_loss,
                loss='binary_crossentropy',
                #loss = hn_multilabel_loss,
                optimizer='adam',
                metrics=['accuracy', FN, FP, TN, TP])

    model.summary()

    ################################################################################
    # Train the model
    ################################################################################
    #early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)                               
    #class_weights = class_weight.compute_sample_weight(class_weight = "balanced", y = train_lbl)
    history = model.fit(train_img, train_lbl,
                            batch_size=batch_size,
                            epochs=num_epochs,
                            verbose=1,
                            #class_weight=class_weights, #{0:5, 1:3, 2:2 ,3:2 ,4:1 ,5:3},
                            validation_split=0.1)
                            #callbacks=[early_stop])
    print(history.history)
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
    plt.savefig('/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/sophia/asparagus/code/get_data/fig_l2.png')
    model.save('/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/sophia/asparagus/code/get_data/l2.h5')

    # convert the history.history dict to a pandas DataFrame   
    hist_df = pd.DataFrame(history.history) 

    # and save to csv
    hist_csv_file = 'history_l2.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)