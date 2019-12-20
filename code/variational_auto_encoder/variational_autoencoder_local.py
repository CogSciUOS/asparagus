'''Example of VAE on MNIST dataset using MLP

The VAE has a modular design. The encoder, decoder and VAE
are 3 models that share weights. After training the VAE model,
the encoder can be used to generate latent vectors.
The decoder can be used to generate MNIST digits by sampling the
latent vector from a Gaussian distribution with mean = 0 and std = 1.

# Reference

[1] Kingma, Diederik P., and Max Welling.
"Auto-Encoding Variational Bayes."
https://arxiv.org/abs/1312.6114
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
from asparagus import *
from PIL import Image

# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def plot_results(models, batch_size=128, model_name="vae_mnist"):
    """Plots labels and MNIST digits as a function of the 2D latent vector

    # Arguments
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    #x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "latent_asparagus.png")
    # display a 30x30 2D manifold of digits
    n = 30

    digit_size = 28
    figure = np.zeros((134 * n, 36 * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(134, 36)
            figure[i * 134: (i + 1) * 134,
                   j * 36: (j + 1) * 36] = digit

    plt.figure(figsize=(10, 10),dpi=200)
    plt.axis('off')
    start_range = digit_size // 2
    end_range = (n - 1) * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.imshow(figure, cmap='gray', aspect=0.8)
    plt.savefig(filename)

    figure -= np.min(figure)
    figure /= np.max(figure)
    figure = np.array(figure*255, dtype=np.uint8)

    im = Image.fromarray(figure)
    im.save("latent_space_raw.png")
    print(im.size)
    #plt.show()


original_dim = 134*36

# network parameters
input_shape = (original_dim, )
intermediate_dim = 512
batch_size = 10
latent_dim = 2
epochs = 2

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = Dense(intermediate_dim, activation='relu')(inputs)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = Dense(original_dim, activation='sigmoid')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')


def variational_loss(reconstruction_loss = "mse"):
    # VAE loss = mse_loss or xent_loss + kl_loss
    if reconstruction_loss == "binary_crossentropy":
        reconstruction_loss = binary_crossentropy(inputs,outputs)
    elif reconstruction_loss == "mse":
        reconstruction_loss = mse(inputs, outputs)

    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    return vae_loss

def train_and_eval(reconstruction_loss="mse", weights=None):
    models = (encoder, decoder)
    data_interface = Asparagus(flatten=True,batch_size=batch_size)

    vae.add_loss(variational_loss(reconstruction_loss))
    vae.compile(optimizer='adam')
    vae.summary()
    plot_model(vae,to_file='vae_mlp.png',show_shapes=True)

    if weights:
        vae.load_weights(weights)
    else:# train the autoencoder
        for epoch in range(epochs):
            train = data_interface.get_training_batches()
            vae.fit_generator(generator = train, steps_per_epoch = len(train.x_train)//batch_size, verbose = 1)#TODO: change to verbose = 2
        vae.save_weights('vae_mlp.h5')

    plot_results(models,batch_size=batch_size,model_name="vae_mlp")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Use mse loss instead of binary cross entropy (default)"
    parser.add_argument("-m","--mse",help=help_, action='store_true')
    args = parser.parse_args()
    if args.mse:
        train_and_eval("mse", args.weights)
    else:
        train_and_eval("binary_crossentropy", args.weights)
