import argparse
import importlib
import os
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
    """defines and parses command line arguments
    """
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
    parser.add_argument('filename_category_1')
    parser.add_argument('filename_category_2')

    return parser.parse_args()


def load_model(model_module, input_shape=None):
    """loads model from file in models folder
    """
    # load model from file
    model_util = importlib.import_module('.' + model_module, 'models')
    # call create_model function of the model_util
    return model_util.create_model(input_shape)


def visualize(x_train, x_test, y_train, y_test, y_pred, model_name=None):
    """Using a PCA to visualize the result

    Arguments:
        x_train(array-like): x data values training
        x_test(array-like): x data values test
        y_train(array-like): y data values training
        y_test(array-like): y data values test
        y_pred(array-like): data values that were predicted
    """
    # PCA
    pca = PCA(2, whiten=True).fit(np.vstack((x_train, x_test)))
    xy_train = pca.transform(x_train)
    xy_test = pca.transform(x_test)

    # color depends on Label
    c_test = y_test[:, 1] + 2 * y_test[:, 0]
    c_train = y_train[:, 1] + 2 * y_train[:, 0]
    c_pred = y_pred[:, 1].round() + 2 * y_pred[:, 0].round()

    # subplots
    fig, ax = plt.subplots(1, 3)
    if model_name is not None:
        fig.suptitle(f'For {model_name}')

    # Train data
    ax[0].set_title('Train data')
    sc_train = ax[0].scatter(*zip(*xy_train), c=c_train,
                             vmin=1, vmax=2, cmap='RdYlGn', alpha=0.2)
    fig.colorbar(sc_train, ax=ax[0])

    # Test data
    ax[1].set_title('Test data')
    sc_test = ax[1].scatter(*zip(*xy_test), c=c_test,
                            vmin=1, vmax=2, cmap='RdYlGn', alpha=0.2)
    fig.colorbar(sc_test, ax=ax[1])

    # Predicted data
    ax[2].set_title('Predicted data')
    sc_pred = ax[2].scatter(*zip(*xy_test), c=c_pred,
                            vmin=1, vmax=2, cmap='RdYlGn', alpha=0.2)
    fig.colorbar(sc_pred, ax=ax[2])

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
        # all columns except auto_width and auto_bended
        drop_columns_starting_with = ['auto_violet', 'auto_blooming', 'auto_length', 'auto_rust_head',
                                      'auto_rust_body', 'is_bruch', 'very_thick',
                                      'thick', 'medium_thick', 'thin', 'very_thin', 'unclassified']

    # drop the columns
    for column in drop_columns_starting_with:
        mask = annotations.columns.str.startswith(column)
        annotations = annotations.loc[:, ~mask]

    # add category to df
    annotations["Label"] = set_label

    return annotations


def load_data(filename_cat_1, filename_cat_2):
    """loading annotationfiles from annotationfolder based on filename.
       Path to annotationfolder can be overwritten by specifying the ANNOTATION_PATH variable.

    Arguments:
        filename_cat_1(string): filename of annotationfile in annotationfolder
        filename_cat_2(string): filename of annotationfile in annotationfolder

    Returns:
        data(dataframe)
    """
    log.info('Loading data')
    data_category_1 = load_annotation(
        os.environ.get("ANNOTATION_PATH", "../../annotations/") + filename_cat_1, drop_columns_starting_with=None, set_label="Category_1")
    data_category_2 = load_annotation(
        os.environ.get("ANNOTATION_PATH", "../../annotations/") + filename_cat_2, drop_columns_starting_with=None, set_label="Category_2")
    data = pd.concat([data_category_1, data_category_2])
    data = pd.get_dummies(data, prefix=['Label'])
    log.info(data.head())
    return data


def main():
    # read arguments from command line
    args = read_arguments()

    data = load_data(args.filename_category_1, args.filename_category_2)
    print(data.describe())

    log.info('Performing train/test-split')
    x = data.iloc[:, :-2].values
    # set Label as y
    y = data[['Label_Category_1', 'Label_Category_2']].values

    # make a train and test split
    # 75% train; 25% test
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)

    log.info('Loading model')
    model = load_model(args.model, input_shape=x_train.shape[1:])

    log.info('Fitting model')
    try:
        # if keras model, make several epochs
        model.fit(x_train, y_train, epochs=20)
    except TypeError:
        model.fit(x_train, y_train)

    if hasattr(model, 'score'):
        score = model.score(x_test, y_test)
        log.info('Score is %.4f', score)

    if hasattr(model, 'evaluate'):
        evaluation = model.evaluate(x_test, y_test)
        log.info('Evaluation is %s', evaluation)

    # get the predictions
    y_pred = model.predict(x_test)

    # Visualize using PCA
    visualize(x_train, x_test, y_train, y_test, y_pred, args.model)


if __name__ == '__main__':
    main()
