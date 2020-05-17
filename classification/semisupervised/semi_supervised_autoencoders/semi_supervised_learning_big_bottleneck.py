import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense, Input
from keras.layers import Conv2D, Flatten, Lambda
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.layers import Lambda, Input, Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Deconvolution2D, Reshape, Concatenate
from scipy import stats
from PIL import Image
import PIL
from keras import backend as K
import tensorflow as tf
import pickle
import argparse
import os
tf.compat.v1.disable_eager_execution()

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

def interlace_unlabeled_batches(x_train,y_train,unlabeled,unlabeled_labels):
    """ Method for interlacing unlabeled batches and labeled batches.
        After application there will be a labeled batch, an unlabeled batch (Labels all -1) etc.
    Args:
        x_train: Training source data
        y_train: Training test data
        unlabeled: Unlabeled training source data
        unlabeled_labels: A tensor filled with value -1 of shape unlabeled.shape
    """
    # Divide into batches
    x_train = np.split(x_train, [batch_size+x*batch_size for x in range(len(x_train)//batch_size)])
    x_train = x_train[:-1]#discard incomplete batch
    y_train = np.split(y_train, [batch_size+x*batch_size for x in range(len(y_train)//batch_size)])
    y_train = y_train[:-1]#discard incomplete batch

    # Divide into batches
    unlabeled = np.split(unlabeled, [batch_size+x*batch_size for x in range(len(unlabeled)//batch_size)])
    unlabeled = unlabeled[:-1]#discard incomplete batch
    unlabeled_labels = np.split(unlabeled_labels, [batch_size+x*batch_size for x in range(len(unlabeled_labels)//batch_size)])
    unlabeled_labels = unlabeled_labels[:-1]#discard incomplete batch

    # Zip them [with_label_batch, without_label_batch, with_label_batch ...]
    min_length = np.min([len(unlabeled_labels),len(y_train)])
    x_train1 = np.array(list(zip(unlabeled[:min_length],x_train[:min_length])))
    y_train1 = np.array(list(zip(unlabeled_labels[:min_length],y_train[:min_length])))

    #Reshape such that there are batch_size times unlabeled examples then batch_time_size labeled examples etc.
    y_train1 = y_train1.reshape([y_train1.shape[0]*y_train1.shape[1],y_train1.shape[2],y_train1.shape[3]])
    y_train1 = y_train1.reshape([y_train1.shape[0]*y_train1.shape[1],y_train1.shape[2]])

    new_shape = [x_train1.shape[0]*x_train1.shape[1]]#Merge first two dimensions
    new_shape.extend(list(x_train1.shape)[2:])
    x_train1 = x_train1.reshape(new_shape)
    new_shape = [x_train1.shape[0]*x_train1.shape[1]]#Merge first two dimensions
    new_shape.extend(list(x_train1.shape)[2:])
    x_train1 = x_train1.reshape(new_shape)
    return x_train1, y_train1

def prepare_data():
    """ Loads and prepares image data and labels. Preparation includes interlacing unlabeled batches and labeled batches.
        This means that there will be: A labeled batch, an unlabeled batch (Labels all -1) etc.
    Returns:
        x_train: Partially labeled training data
        x_test: Completely labeled test data
        y_train: Partially labeled training labels. If labels are missing a value of -1 is assigend.
        y_test: Completely labeled test data.
    """
    #Load labeled data and divide into training and test sets
    labels = pd.read_csv(inpath+"labels.csv",index_col=0)
    selected_labels = ["is_bended","is_violet","has_rost_body","short","thick","thin"]
    labels = labels[selected_labels].values

    images = np.load(inpath+"palette_as_rgb.npy")
    images = images[::3]

    #Padding is required as the net does not work with arbitrary shapes due to problems with reconstructing the same
    #shape in deconvolution layers (Deconvolution increases size by factor 2 or 3 or .. i.e. by an integer factor)
    images = np.pad(images[:,:], pad_width=((0,0),(1,1),(0,0),(0,0)), mode='constant', constant_values=0)
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    images = None#free some memory

    # Load unlabeled and generate array of nan as labels (analogously to x_train, y_train)
    unlabeled = np.load(inpath+"unlabeled.npy")
    unlabeled = unlabeled[::3]
    unlabeled = np.pad(unlabeled[:,:], pad_width=((0,0),(1,1),(0,0),(0,0)), mode='constant', constant_values=0)
    unlabeled = unlabeled.astype('float32') / 255

    unlabeled_labels_shape = [len(unlabeled)]
    unlabeled_labels_shape.extend(list(y_train.shape)[1:])
    unlabeled_labels = np.ndarray(unlabeled_labels_shape)
    unlabeled_labels.fill(-1)

    x_train, y_train = interlace_unlabeled_batches(x_train,y_train,unlabeled,unlabeled_labels)
    return x_train, x_test, y_train, y_test


def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    Args:
        args (tensor): mean and log of variance of Q(z|X)

    Returns:
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def mse(y_true, y_pred):
    """ Mean squared error
    args:
        y_pred: Predicted values
        y_true: True values
    returns:
        Mean squared error
    """
    #y_true = K.print_tensor(y_true, message='y_true = ')
    if not K.is_tensor(y_pred):
        y_pred = K.constant(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)
    return K.mean(K.square(y_pred - y_true), axis=-1)

def check_for_nan(label_inputs):
    """ Is true when one or more values are -1 that indicates no value is present
    args:
        label_inputs: Tensor that potentially contains -1
    returns:
        contains_nan: Tensorflow conditional that indicates wether or not a value is present upon evaluation.
    """
    #label_inputs = K.print_tensor(label_inputs, message='contains_nan = ')
    contains_nan = tf.reduce_any(tf.math.equal(label_inputs,-1))
    #contains_nan = K.print_tensor(contains_nan, message='contains_nan = ')
    return contains_nan

def semi_supervised_autoencoder(filters, kernel_size, latent_dim,bypass_dim, labels_shape, input_shape):
    """ Representation of the semi semi_supervised_autoencoder incuding the respective loss
    Returns:
        Keras model that represents the semi_supervised_autoencoder
    """
    inputs = Input(shape=input_shape, name='encoder_input')
    label_inputs = Input(shape=(labels_shape,), name='label_inputs')

    #encoder
    conv_layer0 = Conv2D(32, kernel_size=(2, 2),activation='relu',input_shape=input_shape)(inputs)
    conv_layer1 = Conv2D(32, (3, 3), activation='relu')(conv_layer0)
    maxpool_layer0 = MaxPooling2D(pool_size=(2, 2))(conv_layer1)
    conv_layer2 = Conv2D(32, (3, 3), activation='relu')(maxpool_layer0)
    maxpool_layer1 = MaxPooling2D(pool_size=(2, 2))(conv_layer2)
    dropout_layer0 = Dropout(0.25)(maxpool_layer1)
    flatten_layer0 = Flatten()(dropout_layer0)
    dense_layer0 = Dense(64, activation='relu')(flatten_layer0)
    x = Dropout(0.5)(dense_layer0)

    label_layer = Dense(labels_shape, name='label_layer')(x)
    label_layer_sigmoid = Lambda(lambda x: tf.sigmoid(x), name='label_layer_sigmoid')(label_layer)
    error_layer = Lambda(lambda x: tf.abs(x[0]-x[1]), name='error_layer')([label_layer_sigmoid,label_inputs])
    dummy_layer = Lambda(lambda x: x*0, name='dummy_layer')(error_layer)#To keep the graph connected

    z = Concatenate()([dummy_layer,  label_layer])#, bypass_layer])

    # decoder
    latent_inputs = Input(shape=(2*labels_shape,), name='z_sampling')

    x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
    x = Reshape((shape[1], shape[2], shape[3]))(x)

    for i in range(2):
        x = Conv2DTranspose(filters=filters,kernel_size=kernel_size,activation='relu',strides=2,padding='same')(x)
        filters //= 2

    outputs = Conv2DTranspose(filters=input_shape[-1], kernel_size=kernel_size, activation='sigmoid',padding='same',name='decoder_output')(x)


    # instantiate encoder model
    encoder = Model([inputs,label_inputs], [z, label_layer_sigmoid], name='encoder')
    encoder.summary()

    # instantiate decoder model
    decoder = Model([latent_inputs], outputs, name='decoder')
    decoder.summary()

    # instantiate VAE model
    outputs = decoder(encoder([inputs,label_inputs])[0])
    vae = Model(inputs=[inputs,label_inputs], outputs=[outputs], name='vae')

    #Loss
    reconstruction_loss = mse(K.flatten(inputs), K.flatten(outputs))
    reconstruction_loss *= input_shape[0] *  input_shape[1] * input_shape[2]
    contains_nan = check_for_nan(label_inputs)
    class_loss = 100+10*K.mean(error_layer)*reconstruction_loss#make the class loss scale with the vae_loss
    combined_loss = class_loss + reconstruction_loss
    final_loss = tf.cond(contains_nan,lambda: reconstruction_loss,lambda: combined_loss)
    vae.add_loss(final_loss)

    return vae

# Hyperparameters
inpath = "/mnt/c/Users/eler/Desktop/asparagus_project/anns_michael/"
outpath = "/mnt/c/Users/eler/Desktop/asparagus_project/anns_michael/"
batch_size = 128
epochs = 20

# Data and shapes
x_train, x_test, y_train, y_test = prepare_data()

input_shape = np.array(x_train.shape)
output_shape = input_shape
shape = np.zeros(4, dtype=np.int32)
shape[0] = -1
shape[-1] = 32
shape[1:3] = (output_shape[1:3]//2)//2#Such that after application of filters the desired output shape is achieved
input_shape = input_shape[1:]
output_shape = output_shape[1:]
labels_shape = y_train.shape[1]

# network parameters
kernel_size = 3
filters = 16
latent_dim = 64
bypass_dim = 2

if __name__ == "__main__":
    vae = semi_supervised_autoencoder(filters, kernel_size, latent_dim,bypass_dim, labels_shape, input_shape)
    vae.compile(optimizer='rmsprop')

    history = vae.fit([x_train,y_train],epochs=epochs,batch_size=batch_size,verbose=1,shuffle=False)#Do not shuffle!!!
    vae.save_weights(outpath+'semi_supervised_big_bottleneck.h5')

    y_test = pd.DataFrame(y_test, columns=["is_bended","is_violet","has_rost_body","short","thick","thin"])
    label_predictions = encoder.predict([x_test,y_test])[-1]

    np.save(outpath+"learning_curve_heads.npy",np.array(history.history["loss"]))
    y_pred = model.predict(x_test)
    conf = get_confusion_matrices(y_pred, y_test,False,True)
    conf[1].round(2).to_csv(outpath+"confusion_heads.csv")
    roc = roc_characteristic(model.predict(x_test), y_test)
    np.save(outpath+"roc_heads.npy",roc)
