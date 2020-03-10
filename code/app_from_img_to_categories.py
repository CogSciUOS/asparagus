""" This is a streamlit app that enables you to select different models
    and inspect the predictions

    Input: - auto values
            - 3 images of an asparagus piece


    Output: - see the predicted feature values
            - and use selected other model to predict categories


    To run the app:
    streamlit run multiple_models_app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow.keras as keras
import tensorflow.keras.models
from sklearn.model_selection import train_test_split


from skimage.io import imread
from skimage.transform import resize

from os import listdir
from io import StringIO
from os.path import isfile, join

from skimage.io import imread

import labelCNN.training as train
import pipeline.train_model as tm

from sklearn.metrics import classification_report


IMAGE_COLUMNS = ['image_a', 'image_b', 'image_c']
AUTO_COLUMNS = ['auto_violet', 'auto_length', 'auto_width', 'auto_bended']
LABEL_COLUMNS = ['is_hollow', 'has_blume', 'has_rost_head',
                 'has_rost_body', 'is_bended', 'is_violet']


@st.cache
def load_feat_data(labels_csv, imagedir):
    raw_feat_data = train.load_df(labels_csv, imagedir)
    # add unique code for combinations of features
    raw_feat_data["Code"] = raw_feat_data[LABEL_COLUMNS].apply(
        convert2binary, axis=1)
    return raw_feat_data


@st.cache
def load_cat_data():
    return tm.load_data("../annotations")


def convert2binary(vector):
    return vector @ 2 ** np.arange(len(vector))


@st.cache
def get_sample(raw_feat_data, sample_idx):
    """ given a sample index of the dataframe return the sample """

    # convert sample to dataset entry
    row = raw_feat_data.iloc[sample_idx]

    resized_shape = (670, 182, 3)
    sample = {'image_a_input': resize(imread(row['image_a']), resized_shape)[np.newaxis].astype(np.float32),
              'image_b_input': resize(imread(row['image_b']), resized_shape)[np.newaxis].astype(np.float32),
              'image_c_input': resize(imread(row['image_c']), resized_shape)[np.newaxis].astype(np.float32),
              'auto_input': row[AUTO_COLUMNS].values[np.newaxis].astype(np.float32) / 300}

    return sample


def predict_features_with_CNN(model, sample):
    return model.predict(sample)


def highlight_diff_vec(data, other, color='pink'):
    """ pandas styler: compare each entry of the dfs (round prediction),
        mark field in prediction red if it not equal"""
    # Define html attribute
    attr = 'background-color: {}'.format(color)
    other = other.transpose()
    # round predicted values to 0 or 1 and compare to true target
    diffs = np.abs((data >= 0.5).values - other.values)
    return pd.DataFrame(np.where(diffs, attr, ''), index=data.index, columns=data.columns)


def evaluate(raw_feat_data, sample_idx, pred_feat_vec):
    """ shows predicted vector and true target and marks differences"""

    pred_feat_vec = pd.DataFrame(
        pred_feat_vec, columns=[LABEL_COLUMNS])

    # highlight differences
    st.table(pred_feat_vec.style.apply(highlight_diff_vec, axis=None, other=pd.DataFrame(
        raw_feat_data[LABEL_COLUMNS].iloc[sample_idx])).format("{:.2f}"))

    "The input/true target vector was/is:"
    true_vec = pd.DataFrame(
        raw_feat_data[LABEL_COLUMNS+AUTO_COLUMNS].iloc[sample_idx]).transpose()
    st.table(true_vec)


def streamlit_model_summary(model_file, model):
    """ display model summary as markdown """
    with StringIO() as s:
        def w(l):
            s.write(l)
            s.write('\n')
        model.summary(print_fn=w)
        s.seek(0)
        summary = s.read()
    st.markdown(f'```\n{summary}\n```')


def main():
    st.title('Asparagus label prediction')

    imgdirs = ["/home/katha/labeled_images",
               "/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/without_background_pngs/"]
    imgdir_option = st.selectbox(
        'Please select the image directory',
        imgdirs)

    csv_files = ["all_label_files.csv",
                 "/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/katha/labels.csv"]
    csv_option = st.selectbox(
        'Please select the csv file',
        csv_files)

    '# Generated dataframe'
    raw_feat_data = load_feat_data(csv_option, imgdir_option)

    if st.checkbox('Show training dataframe'):
        raw_feat_data
    "The dataframe has the shape", raw_feat_data.shape, "."

    num_feat = len(AUTO_COLUMNS) + len(LABEL_COLUMNS)
    num_constellations = 2**len(LABEL_COLUMNS)
    "There are", num_feat, "different features and", 3, "images for each piece."
    "We want to predict the", len(
        LABEL_COLUMNS), "manually labeled features for unseen pieces."
    "Every feature is either 0 or 1."
    "Which makes", num_constellations, "unique constellations of features that we could predict."

    "## How the individual features are distributed"
    count_feat = [{col: raw_feat_data[col].sum()} for col in LABEL_COLUMNS]
    st.bar_chart(count_feat)

    "## How the constellations are distributed"
    st.bar_chart(raw_feat_data['Code'].value_counts())

    "# Choose the model that you want to train"
    model_folder = "labelCNN/models/"
    models = [model[:model.index(".")] for model in listdir(
        model_folder) if isfile(join(model_folder, model))]

    model_option = st.selectbox(
        'Which model do you want to train?',
        models)
    model_path = model_folder + model_option + ".h5"

    # load selected model
    CNN_model = tensorflow.keras.models.load_model(model_path)

    # show information about the model
    streamlit_model_summary(model_path, CNN_model)

    "# Inspect the dataset"
    "Select an image"
    sample_idx = st.slider('Sample', 0, int(raw_feat_data.shape[0]), value=0)
    selected_sample = pd.DataFrame(
        raw_feat_data[LABEL_COLUMNS+AUTO_COLUMNS].iloc[sample_idx]).transpose()
    st.write(selected_sample)

    # TODO hier auslagern
    img_pathes = raw_feat_data[IMAGE_COLUMNS].iloc[sample_idx]

    # display the images
    images_imread = [imread(img_path) for img_path in img_pathes]
    # stack and display 3 images
    st.image(np.hstack([np.squeeze(i)
                        for i in images_imread]), use_column_width=True)

    if st.checkbox('Make prediction for feature vector'):

        "The model predicts the following target vector:"
        sample = get_sample(raw_feat_data, sample_idx)

        pred_feat_vec = predict_features_with_CNN(CNN_model, sample)

        # show predicted and true target vector and mark differences
        evaluate(raw_feat_data, sample_idx, pred_feat_vec)

        # TODO
        # calculate different losses

    if st.checkbox('Make category prediction for feature prediction'):

        "# From features to categories"
        raw_cat_data, dummy_data = load_cat_data()
        labels = [col for col in dummy_data if col.startswith('Class')]

        # display the train df
        if st.checkbox('Show category training dataframe'):
            dummy_data

            "There are", len(labels), "labels."
            "We want to learn the following labels: "
            # draw histogram to see how classes are distributed
            st.bar_chart(raw_cat_data['Class'].value_counts())

        ######
        # train the model
        x = dummy_data.iloc[:, :-len(labels)].values
        # set Label as y
        y = dummy_data[labels].values

        # make a train and test split: 75% train; 25% test
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, random_state=1)

        "After we perform the train/test-split, we have", len(
            x_train), "training samples"
        "After we perform the train/test-split, we have", len(
            x_test), "test samples"

        "## Choose the model that you want to train"
        "Right now there is a random forest model and a multilayer preceptron."

        model_folder = "pipeline/models"
        models = [model[:model.index(".")] for model in listdir(
            model_folder) if isfile(join(model_folder, model))]

        model_option = st.selectbox(
            'Which model do you want to train?',
            models)

        # load selected model
        model = tm.load_model(model_option, input_shape=x_train.shape[1:])

        '## Fitting the model'
        try:
            # if keras model, make several epochs
            model.fit(x_train, y_train, epochs=500)
        except TypeError:
            model.fit(x_train, y_train)

        if hasattr(model, 'score'):
            score = model.score(x_test, y_test)
            st.write('Score on validation dataset is', np.round(score, 3))

        if hasattr(model, 'evaluate'):
            evaluation = model.evaluate(x_test, y_test)
            st.write('Evaluation is', evaluation)

        # get the predictions
        y_pred = model.predict(x_test)

        "Number of samples in validation set:", len(y_pred)
        ######

        "## Results"

        "### Classification report"
        if st.checkbox('Show classification report'):
            st.write(pd.DataFrame(classification_report(y_test.argmax(axis=1),
                                                        y_pred.argmax(axis=1), target_names=labels, output_dict=True)).transpose())

        "### Confusion Matrix"
        if st.checkbox('Show confusion matrix'):
            conf_matrix, ax = tm.conf_matrix(
                x_train, x_test, y_train, y_test, y_pred, labels, model_option)
            st.pyplot()

        "#### Selected sample"

        "Looking at the", sample_idx, "th feature prediction"

        # TODO
        # lieber in gecachte function
        # pred_feat_vec = predict_features(model, raw_feat_data, sample_idx)

        # TODO is this necessary here?
        sample = get_sample(raw_feat_data, sample_idx)
        pred_feat_vec = predict_features_with_CNN(CNN_model, sample)
        pred_feat_vec = pd.DataFrame(pred_feat_vec, columns=[
                                     LABEL_COLUMNS]).round(0)
        auto_vals = pd.DataFrame(
            raw_feat_data[AUTO_COLUMNS].iloc[sample_idx]).transpose().reset_index(drop=True)
        full_feat_vec = pd.concat(
            [pred_feat_vec, auto_vals], axis=1, ignore_index=True)
        full_feat_vec.columns = LABEL_COLUMNS+AUTO_COLUMNS

        "Predicted feature vector concatenated with auto values"
        st.write(full_feat_vec)

        # convert
        row_values = full_feat_vec.values

        # predict
        cat_prediction = model.predict(row_values)

        st.write("feature vector", str(row_values))
        str.write(" has prediction ", str(cat_prediction))

        st.write("Use argmax to get label: the category label is:")
        st.write(str(cat_prediction.argmax(axis=1)))


if __name__ == '__main__':
    main()
