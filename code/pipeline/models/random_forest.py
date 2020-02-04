import numpy as np
from sklearn.ensemble import RandomForestClassifier


class OneHotRandomForestClassifier(RandomForestClassifier):

    def fit(self, x, y):
        self.number_of_classes = y.shape[1]
        # convert one hot encoding to numeric class labels
        y = np.argmax(y, axis=1)
        return super().fit(x, y)

    def predict(self, x):
        """convert back to one-hot encoding
        """
        y = super().predict(x)
        y_new = np.zeros((y.shape[0], self.number_of_classes))
        y_new[np.arange(y.shape[0]), y] = 1
        return y_new


def create_model(input_shape=None):
    return OneHotRandomForestClassifier()
