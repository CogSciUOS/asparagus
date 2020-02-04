import argparse
import glob
import importlib
import os
from pathlib import Path
import textwrap
from logging import getLogger, StreamHandler, INFO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
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
    parser.add_argument(
        'foldername', help='foldername in which the annotation csv files with class lie')

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

    """
    print()
    print("Classification report (one hot encoding):")
    print(classification_report(y_test, y_pred))
    """

    print("Classification report (encoding with labels):")
    print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1)))

    # confusion matrix
    # The matrix output by sklearn's confusion_matrix() is such that
    # C_{i, j} is equal to the number of observations known to be in group i but predicted to be in group j
    conf_mat = confusion_matrix(
        y_test.argmax(axis=1), y_pred.argmax(axis=1))

    seaborn.heatmap(conf_mat)
    plt.show()

    """
    from confusion_matrix import plot_confusion_matrix_from_data
    columns = []
    annot = True
    cmap = 'Oranges'
    fmt = '.2f'
    lw = 0.5
    cbar = False
    show_null_values = 2
    pred_val_axis = 'y'
    # size::
    fz = 2
    figsize = [5, 5]
    if(len(y_test) > 10):
        fz = 2
        figsize = [5, 5]
    plot_confusion_matrix_from_data(y_test.argmax(axis=1), y_pred.argmax(axis=1), columns,
                                    annot, cmap, fmt, fz, lw, cbar, figsize, show_null_values, pred_val_axis)
    """

    # PCA
    pca = PCA(2, whiten=True).fit(np.vstack((x_train, x_test)))
    xy_train = pca.transform(x_train)
    xy_test = pca.transform(x_test)

    # color depends on Label
    binary_code = 2 ** np.arange(y_train.shape[1])
    c_test = np.log2(y_test @ binary_code + 1)
    c_train = np.log2(y_train @ binary_code + 1)
    c_pred = np.log2(y_pred.round() @ binary_code + 1)

    # subplots
    params = dict(cmap='gist_ncar', alpha=0.6)
    fig, ax = plt.subplots(2, 2, figsize=(20, 10))
    ax = ax.flatten()
    fig.delaxes(ax[3])
    if model_name is not None:
        fig.suptitle(f'For {model_name}')

    # Train data
    ax[0].set_title('Train data')
    sc_train = ax[0].scatter(*zip(*xy_train), c=c_train, **params)
    ax[0].set_ylim([-.5, 5])
    ax[0].set_xlim([-.5, 5])
    fig.colorbar(sc_train, ax=ax[0])

    # Test data
    ax[1].set_title('Test data')
    sc_test = ax[1].scatter(*zip(*xy_test), c=c_test, **params)
    ax[1].set_ylim([-.5, 5])
    ax[1].set_xlim([-.5, 5])
    fig.colorbar(sc_test, ax=ax[1])

    # Predicted data
    ax[2].set_title('Predicted data')
    sc_pred = ax[2].scatter(*zip(*xy_test), c=c_pred, **params)
    ax[2].set_ylim([-.5, 5])
    ax[2].set_xlim([-.5, 5])
    fig.colorbar(sc_pred, ax=ax[2])

    plt.show()


def load_annotation(filename, drop_columns_starting_with=None):
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
    annotations = pd.read_csv(filename, delimiter=";", index_col=0)

    # remove unclassified it was only 2 rows for the labeled folders
    unclassifiable = annotations[annotations['unclassified'] == 1].index
    annotations.drop(unclassifiable, inplace=True)

    if drop_columns_starting_with is None:
        # all autogenerated columns except auto_width, auto_bended, auto_violet, auto_length
        drop_columns_starting_with = ['auto_blooming', 'auto_rust_head',
                                      'auto_rust_body', 'is_bruch', 'very_thick',
                                      'thick', 'medium_thick', 'thin', 'very_thin', 'unclassified']

    # drop the columns
    for column in drop_columns_starting_with:
        mask = annotations.columns.str.startswith(column)
        annotations = annotations.loc[:, ~mask]

    # drop columns with NaN (in labeled folder 32 rows)
    annotations = annotations.dropna()

    return annotations


def load_data(folder):
    """loading annotationfiles from annotationfolder based on filename.
       Path to annotationfolder can be overwritten by specifying the ANNOTATION_PATH variable.

    Arguments:
        filename_cat_1(string): filename of annotationfile in annotationfolder
        filename_cat_2(string): filename of annotationfile in annotationfolder

    Returns:
        data(dataframe)
    """
    log.info('Loading data')

    header_written = False
    with Path("concatenated_annotations_with_class.csv").open('w') as outf:
        for infile in Path(folder).iterdir():
            if not infile.is_file():
                continue
            infile_name = infile.name[infile.name.index("_"):]
            infile_cat = infile_name[7:-4]
            lines = infile.read_text().splitlines()

            if header_written:
                cat_added = [line + ";" + infile_cat for line in lines]
                lines = cat_added[1:]

            else:
                lines[0] = lines[0].replace(",,", "")
                lines[0] = lines[0] + ";Class"
                for i in range(1, len(lines)):
                    lines[i] = lines[i] + ";" + infile_cat
                header_written = True

            outf.write('\n'.join(lines) + '\n')

    data = load_annotation("concatenated_annotations_with_class.csv")
    data = pd.get_dummies(data, columns=["Class"], prefix=['Class'])

    return data


def main():
    # read arguments from command line
    args = read_arguments()

    # data = load_data(args.filename_category_1, args.filename_category_2)
    data = load_data(args.foldername)

    # Labels
    labels = [col for col in data if col.startswith('Class')]
    log.info(labels)
    log.info("Number of labels:")
    log.info(len(labels))

    log.info('Performing train/test-split')
    x = data.iloc[:, :-len(labels)].values
    # set Label as y
    y = data[labels].values

    # make a train and test split
    # 75% train; 25% test
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)

    log.info("Number of samples in train:")
    log.info(len(x_train))
    log.info("Number of samples in test:")
    log.info(len(x_test))

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

    log.info("Number of samples in predictions:")
    log.info(len(y_pred))

    print("feature vector", y_test[0])
    print("has prediction")
    print("feature vector", y_pred[0].round(3))

    print("Use argmax to get labels")
    print("sample label", y_test.argmax(axis=1)[0])
    print("has prediction")
    print("sample label", y_pred.argmax(axis=1)[0])

    # Visualize using confusion matrix ands PCA
    visualize(x_train, x_test, y_train, y_test, y_pred, args.model)


if __name__ == '__main__':
    main()
