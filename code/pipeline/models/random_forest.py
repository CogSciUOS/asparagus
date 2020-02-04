import numpy as np
from sklearn.ensemble import RandomForestClassifier


class OneHotRandomForestClassifier(RandomForestClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.number_of_classes = None

    def fit(self, x, y, *args, **kwargs):
        self.number_of_classes = y.shape[1]
        y = np.argmax(y, axis=1)
        return super().fit(x, y, *args, **kwargs)

    def predict(self, x, *args, **kwargs):
        y = super().predict(x, *args, **kwargs)
        y_new = np.zeros((y.shape[0], self.number_of_classes), dtype=np.uint8)
        ind = np.vstack([np.arange(y.shape[0]), y])
        y_new[ind[0], ind[1]] = 1
        return y_new


def create_model(input_shape=None):
    return OneHotRandomForestClassifier()
