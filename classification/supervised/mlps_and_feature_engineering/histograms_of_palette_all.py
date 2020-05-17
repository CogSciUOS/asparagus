import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def roc_characteristic(y_pred,y_test):
    fpr = []
    tpr = []
    biases = []
    for bias in np.linspace(-50,50,1000):
        summary, _ = get_confusion_matrices(y_pred,y_test,bias=np.abs(sigmoid(bias)))
        fpv = summary["False positive"]/(summary["False positive"]+summary["True negative"])
        tpv = summary["True positive"]/(summary["True positive"]+summary["False negative"])
        biases.append(bias)
        fpr.append(fpv)
        tpr.append(tpv)
    return np.array([np.array(fpr), np.array(tpr)])

def get_confusion_matrices(y_pred,y_test,bias=False, additional_measures=False):
    #Calculate confusion matrix
    if bias == False:
        bias = .5
    y_pred = y_pred > bias
    y_test = pd.DataFrame(y_test, dtype=np.int32)
    y_pred = np.array(y_pred, dtype=np.int32)
    y_test = pd.DataFrame(y_test)
    false_negatives = np.sum(np.logical_and(y_test == 1,y_pred==0),axis=0)#y_test is 1 while y_pred did not indicate
    false_positives = np.sum(np.logical_and(y_test == 0,y_pred==1),axis=0)#y_test is 0 while y_pred say it was 1
    true_positives = np.sum(np.logical_and(y_test == 1,y_pred==1),axis=0)#both indicate 1
    true_negatives = np.sum(np.logical_and(y_test == 0,y_pred==0),axis=0)#both indicate 0
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


inpath = "/mnt/c/Users/eler/Desktop/asparagus_project/anns_michael/"
outpath = "/mnt/c/Users/eler/Desktop/asparagus_project/anns_michael/"
batch_size = 10
epochs = 500

if __name__ == "__main__":
    images = np.load(inpath+"palette_whole.npy")
    histograms = []
    for i, img in enumerate(images):
        if i % 100 == 0:
            print('.',end='')
        flat = img.flatten()
        foreground_pixels = flat[flat!=254.0]
        hist = np.histogram(foreground_pixels,bins=np.arange(256),density=True)
        histograms.append(hist[0])
    histograms = np.array(histograms)
    histograms = np.expand_dims(np.expand_dims(histograms,1),3)
    labels = pd.read_csv(inpath+"labels.csv",header=0)

    selected_labels = ["is_violet","has_rost_body"]
    labels = labels[selected_labels]

    x_train, x_test, y_train, y_test = train_test_split(histograms, labels, test_size=0.2, random_state=42)

    sample_histogram = x_train[0]
    sample_targets = np.array(y_train)[0]
    num_classes = sample_targets.shape[0]
    img_rows, img_cols = list(sample_histogram.shape)[0:2]
    input_shape = [img_rows, img_cols,1]

    model = Sequential()
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='sigmoid'))
    model.compile(loss=keras.losses.binary_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])

    history = model.fit(x_train, y_train.values,batch_size=batch_size,epochs=epochs,verbose=2,validation_data=(x_test, y_test.values))
    score = model.evaluate(x_test, y_test.values, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    np.save(outpath+"learning_curve_histograms.npy",np.array(history.history["loss"]))
    y_pred = model.predict(x_test)
    conf = get_confusion_matrices(y_pred, y_test,False,True)
    conf[1].round(2).to_csv(outpath+"confusion_histograms.csv")
    roc = roc_characteristic(model.predict(x_test), y_test)
    np.save(outpath+"roc_histograms.npy",roc)
