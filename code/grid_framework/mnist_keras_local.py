import tensorflow as tf
import os
import argparse
from grid import typecast
import sys


def train_keras(n_neurons, optimizer, epochs):
    """ Trains a MLP on the mnist dataset. Uses the specified optimizer and number of fully connected layers.
    Args:
        n_layers:  Number of neurons in the fully connected layer of the MLP.
        optimizer: Gradient descent optimizer.
    """

    mnist = tf.keras.datasets.mnist

    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(n_neurons, activation=tf.nn.relu),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=epochs, verbose=2)
    print("\n The performance in evaluation for " + str(n_neurons) +" neurons was: \n")
    print(model.evaluate(x_test, y_test,verbose=0))

if __name__ == "__main__":
    train_keras(*typecast(sys.argv[1:]))
