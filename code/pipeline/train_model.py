import argparse
import importlib
import textwrap
from logging import getLogger, StreamHandler, INFO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


log = getLogger(__file__)
log.setLevel(INFO)
log.addHandler(StreamHandler())


def read_arguments():
    parser = argparse.ArgumentParser()
    models_help = '''
    Model name which refers to the python file. E.g. random_forest refers to
    models/random_forest.py.

    The referenced script must have a function:

        def create_model(input_shape=None):
            ...
            return model

    and the resulting model must provide the functions

        def fit(self, x, y):
            ...

        def predict(self, x):
            ...
            return prediction

    And zero or more of the following functions:

        def score(self, x, y):
            ...
            return score

        def evaluate(self, x, y):
            ...
            return evaluation

    Most keras and scikit-learn models already provide the first two methods
    and either the score (scikit-learn) or evaluate (keras) methods.'''

    parser.add_argument('model', help=textwrap.dedent(models_help))
    parser.add_argument('dataset', help='File path to the dataset, e.g. data/random_data.csv')
    
    return parser.parse_args()


def load_model(model_module, input_shape=None):
    importlib.invalidate_caches()
    model_util = importlib.import_module('.' + model_module, 'models')
    return model_util.create_model(input_shape)


def visualize(x_train, x_test, y_train, y_test, y_pred):
    """Using a PCA to visualize the result
    
    Arguments:
        x_train {[type]} -- [description]
        x_test {[type]} -- [description]
        y_train {[type]} -- [description]
        y_test {[type]} -- [description]
        y_pred {[type]} -- [description]
    """
    pca = PCA(2, whiten=True).fit(np.vstack((x_train, x_test)))
    xy_plot = pca.transform(x_test)

    # Test data
    fig, ax = plt.subplots(1, 2)
    ax[0].set_title('Test data')
    sc_test = ax[0].scatter(*zip(*xy_plot), c=y_test, vmin=0, vmax=1, cmap='RdYlGn')
    ax[0].set_aspect('equal')
    fig.colorbar(sc_test, ax=ax[0])

    # Predicted data
    ax[1].set_title('Predicted data')
    sc_pred = ax[1].scatter(*zip(*xy_plot), c=y_pred, vmin=0, vmax=1, cmap='RdYlGn')
    ax[1].set_aspect('equal')
    fig.colorbar(sc_pred, ax=ax[1])

    plt.waitforbuttonpress()

def load_annotation(filename_1, drop_columns_starting_with=None, set_label=None):
    """This functions loads the annotation label csv file
    And sets category if specified.

    The first columns are used as index columns.

    Args:
        drop_columns_starting_with(list): Columns starting with these strings will be dropped.
                                          If it is `None`, it defaults to
                                          ['auto', 'is_bruch', 'very_thick', 'thick', 'medium_thick', 'thin', 'very_thin', 'unclassified'].
        set_category(str):                Add a category if possible

    Returns:
        The dataframe
    """
    annotations = pd.read_csv(filename_1, delimiter=";", index_col=0)

    if drop_columns_starting_with is None:
        drop_columns_starting_with = ['auto', 'is_bruch', 'very_thick',
                                      'thick', 'medium_thick', 'thin', 'very_thin', 'unclassified']

    # drop the columns
    for column in drop_columns_starting_with:
        mask = annotations.columns.str.startswith(column)
        annotations = annotations.loc[:, ~mask]

    # add category to df
    annotations["Label"]= set_label

    return annotations

def dummy_labels(df):
    return pd.get_dummies(df,prefix=['Label'])


def main():
    # read arguments from command line
    args = read_arguments()

    log.info('Loading data')
    data = pd.read_csv(args.dataset)
    #data_Anna = load_annotation("data/1A_Anna.csv", drop_columns_starting_with=None, set_label="1A_Anna")
    #data_Bona = load_annotation("data/1A_Bona.csv", drop_columns_starting_with=None, set_label="1A_Bona")
    #data = pd.concat([data_Anna, data_Bona])
    #data = dummy_labels(data)
    log.info(data.head())

    log.info('Performing train/test-split')
    # ignore index and label
    x = data.iloc[:, 1:-1].values
    # set Label as y
    # y = data['Label_1A_Anna'].values
    y = data['Label'].values
    # make a train and test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)

    log.info('Loading model')
    model = load_model(args.model, input_shape=x_train.shape[1:])

    log.info('Fitting model')
    model.fit(x_train, y_train)


    if hasattr(model, 'score'):
        score = model.score(x_test, y_test)
        log.info('Score is %.4f', score)
    
    if hasattr(model, 'evaluate'):
        evaluation = model.evaluate(x_test, y_test)
        log.info('Evaluation is %s', evaluation)

    # get the predictions
    y_pred = model.predict(x_test).flatten()

    # Visualize using PCA
    visualize(x_train, x_test, y_train, y_test, y_pred)


if __name__ == '__main__':
    main()
