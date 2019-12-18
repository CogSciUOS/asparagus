import string

import numpy as np
import pandas as pd


if __name__ == '__main__':
    Nfeatures = 24
    Nsamples = 739

    probabilities = np.random.random(Nfeatures)
    print(probabilities)
    random_values = np.asarray(np.random.random((Nsamples, Nfeatures)) < probabilities, dtype=np.int)
    print(random_values)
    labels = [f'Feature{l}' for l in string.ascii_uppercase[:Nfeatures - 1]] + ['Label']
    df = pd.DataFrame(random_values, columns=labels)
    df.to_csv('data/random_data.csv')
    print(df.head())
