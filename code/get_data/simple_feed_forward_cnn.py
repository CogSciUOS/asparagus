from __future__ import print_function
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from grid import *

def hstack(images):
    images1 = np.ndarray([images.shape[0]//3,images.shape[1],images.shape[2]*images.shape[3],images.shape[3]])
    for idx in range(len(images1)):
        if (idx % 100) == 0:
            print(".",end = "")
        images1[idx,:,:,:] = np.hstack([np.hstack([images[idx*3],images[(idx*3)+1]]),images[(idx*3)+2]])
    return images1

def simple_cnn(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(2, 2),activation='relu',input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='sigmoid'))

    model.compile(loss=keras.losses.binary_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])
    return model


def get_confusion_matrices(y_pred,y_test,bias=False):
    #Calculate confusion matrix
    if bias == False:
        bias = .5
    y_pred = y_pred > bias
    y_test = pd.DataFrame(y_test, dtype=np.int32)
    y_pred = np.array(y_pred, dtype=np.int32)
    y_test = pd.DataFrame(y_test)
    false_negatives = pd.DataFrame(np.sum(y_test>y_pred,axis=0))#y_test is 1 while y_pred did not indicate
    false_positives = np.sum(y_test<y_pred,axis=0)#y_test is 0 while y_pred say it was 1
    true_positives = np.sum(pd.DataFrame((y_pred+y_test) == 2.0,dtype=np.int32),axis=0)#both indicate 1
    true_negatives = np.sum((y_pred+y_test) == 0.0,axis=0)#both indicate 0
    summary = pd.DataFrame()
    summary['False positive'] = false_positives
    summary['False negative'] = false_negatives
    summary['True positive'] = true_positives
    summary['True negative'] = true_negatives
    summary_percent = (summary/summary.sum(axis=1)[0])*100
    return summary, summary_percent

def roc_characteristic(y_pred,y_test):
    false_positives = []
    true_positives = []
    real_positive = y_test.sum(axis=0)
    real_negative = len(y_test)-y_test.sum(axis=0)
    biases = []
    for bias in range(0,100):
        summary, _ = get_confusion_matrices(y_pred,y_test,bias=bias/100)
        false_positive = summary["False positive"]/real_negative
        true_positive = summary["True positive"]/real_positive
        biases.append(bias)
        false_positives.append(false_positive.values)
        true_positives.append(true_positive.values)
    return np.array([np.array(false_positives), np.array(true_positives)]), biases

def train_and_eval(all_perspectives = True):
    images = np.load("palette_as_rgb.npy")
    imgs = images.copy()
    if all_perspectives:
        print("Train on all perspectives")
        images = hstack(images)
    else:
        print("Train on single perspective")
        images = images[::3]
    input_shape = list(images.shape[1:])
    labels = pd.read_csv("labels.csv",header=0)
    labels = labels[["is_bended","is_violet","short","thick","thin"]]
    labels
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    #Input shape
    sample_img = x_train[0]
    sample_targets = np.array(y_train)[0]
    num_classes = sample_targets.shape[0]
    img_rows, img_cols = list(sample_img.shape)[0:2]

    # Hyperparameters
    batch_size = 10
    epochs = 100

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    model = simple_cnn(input_shape,num_classes)
    history = model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    y_pred = model.predict(x_test)
    summary, summary_percent = get_confusion_matrices(y_pred,y_test,bias=.5)
    print("\nSummary")
    print(summary)
    summary_percent.to_csv("confusion_histograms_of_palette.csv")

    print("\nSummary percent")
    print(summary_percent)

    print("\nSummary percent with bias added\n\n")
    summary, summary_percent = get_confusion_matrices(y_pred,y_test,bias=.25)
    print(summary_percent)
    if all_perspectives:
        summary_percent.to_csv("confusion_matrix_simple_cnn.csv")
    else:
        summary_percent.to_csv("confusion_matrix_simple_cnn_single_perspective.csv")

    roc = roc_characteristic(y_pred,y_test)
    if all_perspectives:
        np.save("roc_simple_cnn.npy",roc[0])
    else:
        np.save("roc_simple_cnn_single_perspective.npy",roc[0])


if __name__ == "__main__":
    args = typecast(sys.argv[1:])
    train_and_eval(args[0])
