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
from scipy import stats
from PIL import Image
import PIL

def sigmoid(x):
    """ Sigmoid function
    Args:
        x: The input tensor
    Returns:
        sigmoid transform of x
    """
    return 1 / (1 + np.exp(-x))

def roc_characteristic(y_pred,y_test):
    """ Receiver operating characteristic for the specified data
    Args:
        y_pred: Predictions (tensor)
        y_test: True values (tensor)
    """
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
    """ Calculates confusion metrix containing true and false positives / negatives as well as potentially additional_measures
    Args:
        y_pred: Predicted labels
        y_test: True labels
        bias: An integer between 0 and 1 or False. If false a value of .5 is assigned. The bias defines the threshold used to binarize output layer activations.
        additional_measures: Boolean that indicates whether or not additional measures shall be computed (Sensitivity and Specificity)
    """
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


def rotate_heads(heads):
    """ Rotates images such that foreground pixels are as close as possible to a vertical line
    Args:
        heads: Images of heads
    Returns:
        Rotated images
    """
    #Rotate heads
    for i, img in enumerate(heads):
        if i % 100 == 0:
            print(".", end = "")
        try:
            slope = stats.linregress(np.where(heads[i,:,:,0]!=0))[0]
            slope = -np.rad2deg(slope)
            img = np.array(Image.fromarray(np.array(img, dtype=np.uint8)).rotate(slope, PIL.Image.BICUBIC))
            horizontal_center = np.argmax(np.mean(img[:,:,0],0))
            shift = horizontal_center-32
            img = np.pad(img, [(0,0),(64,64),(0,0)],"constant")
            img = img[:,64+shift:2*64+shift,:]
            heads[i,:,:,:] = img
        except:
            pass
    return heads

def hstack(images):
    """ Stacks images horizontally
    args:
        images: Tensor of images
    returns:
        Rotated images
    """
    images1 = np.ndarray([images.shape[0]//3,images.shape[1],images.shape[2]*images.shape[3],images.shape[3]])
    for idx in range(len(images1)):
        if (idx % 100) == 0:
            print(".",end = "")
        images1[idx,:,:,:] = np.hstack([np.hstack([images[idx*3],images[(idx*3)+1]]),images[(idx*3)+2]])
    return images1


inpath = "/mnt/c/Users/eler/Desktop/asparagus_project/anns_michael/"
outpath = "/mnt/c/Users/eler/Desktop/asparagus_project/anns_michael/"
batch_size = 10
epochs = 40

if __name__ == "__main__":
    heads = np.load(inpath+"heads.npy")
    heads = rotate_heads(heads)
    heads = hstack(heads)

    input_shape = list(heads.shape[1:])
    labels = pd.read_csv(inpath+"labels.csv",header=0)
    labels = labels[["has_blume","has_rost_head"]]
    images = heads

    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    sample_img = x_train[0]
    sample_targets = np.array(y_train)[0]

    num_classes = sample_targets.shape[0]
    img_rows, img_cols = list(sample_img.shape)[0:2]

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

    history = model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    np.save(outpath+"learning_curve_heads.npy",np.array(history.history["loss"]))
    y_pred = model.predict(x_test)
    conf = get_confusion_matrices(y_pred, y_test,False,True)
    conf[1].round(2).to_csv(outpath+"confusion_heads.csv")
    roc = roc_characteristic(model.predict(x_test), y_test)
    np.save(outpath+"roc_heads.npy",roc)
