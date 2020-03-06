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

from skimage.io import imread

from os import listdir
from io import StringIO
from os.path import isfile, join

import labelCNN.training as train


IMAGE_COLUMNS = ['image_a', 'image_b', 'image_c']
AUTO_COLUMNS = ['auto_violet', 'auto_length', 'auto_width', 'auto_bended']
LABEL_COLUMNS = ['is_hollow', 'has_blume', 'has_rost_head',
                 'has_rost_body', 'is_bended', 'is_violet']


@st.cache
def load_data(labels_csv, imagedir):
    raw_data = train.load_df(labels_csv, imagedir)
    # add unique code for combinations of features
    raw_data["Code"] = raw_data[LABEL_COLUMNS].apply(convert2binary, axis=1)
    return raw_data


def convert2binary(vector):
    return vector @ 2 ** np.arange(len(vector))


def predict_features(model, raw_data, sample_idx):

    # convert sample to dataset entry
    df_sample = raw_data.iloc[sample_idx:sample_idx+1]
    # .to_frame().drop(labels=["Code"]).transpose()

    train_dataset, _ = train.create_dataset(
        df_sample, batch_size=1)

    dataset_sample = train_dataset.take(1)

    return model.predict(dataset_sample)


def highlight_diff(s):
    '''
    highlight the maximum in a Series yellow.
    '''
    is_max = s > 10
    return ['background-color: yellow' if v else '' for v in is_max]


def evaluate(true_labels, predicted_labels):
    pass


def streamlit_model_summary(model_file, model):
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

    imgdirs = ["/home/katha/labeled_images"]
    imgdir_option = st.selectbox(
        'Please select the image directory',
        imgdirs)

    csv_files = ["all_label_files.csv"]
    csv_option = st.selectbox(
        'Please select the csv file',
        csv_files)

    '# Generated dataframe'
    raw_data = load_data(csv_option, imgdir_option)

    if st.checkbox('Show training dataframe'):
        raw_data
    "The dataframe has the shape", raw_data.shape, "."

    num_feat = len(AUTO_COLUMNS) + len(LABEL_COLUMNS)
    num_constellations = 2**len(LABEL_COLUMNS)
    "There are", num_feat, "different features and", 3, "images for each piece."
    "We want to predict the", len(
        LABEL_COLUMNS), "manually labeled features for unseen pieces."
    "Every feature is either 0 or 1."
    "Which makes", num_constellations, "unique constellations of features that we could predict."

    "## How the individual features are distributed"
    count_feat = [{col: raw_data[col].sum()} for col in LABEL_COLUMNS]
    st.bar_chart(count_feat)

    "## How the constellations are distributed"
    st.bar_chart(raw_data['Code'].value_counts())

    "# Choose the model that you want to train"
    model_folder = "labelCNN/models/"
    models = [model[:model.index(".")] for model in listdir(
        model_folder) if isfile(join(model_folder, model))]

    model_option = st.selectbox(
        'Which model do you want to train?',
        models)
    model_path = model_folder + model_option + ".h5"

    # load selected model
    model = tensorflow.keras.models.load_model(model_path)

    # show information about the model
    streamlit_model_summary(model_path, model)

    "# Inspect the dataset"
    "Select an image"
    sample_idx = st.slider('Sample', 0, int(raw_data.shape[0]), value=0)
    img_pathes = raw_data[IMAGE_COLUMNS].iloc[sample_idx]

    # display the images
    images_imread = [imread(img_path) for img_path in img_pathes]
    # stack and display 3 images
    st.image(np.hstack([np.squeeze(i)
                        for i in images_imread]), use_column_width=True)

    "The model predicts the following target vector:"
    #pred_feat_vec = predict_features(model, raw_data, sample_idx)
    st.write(pred_feat_vec)
    pred_feat_vec = pd.DataFrame([1, 1, 8, 1, 11, 1], index=[
                                 LABEL_COLUMNS]).transpose().style.apply(highlight_diff)
    st.table(pred_feat_vec)

    f"The input vector / the true target vector is:"
    feat_vec = pd.DataFrame(
        raw_data[LABEL_COLUMNS+AUTO_COLUMNS].iloc[sample_idx]).transpose()
    st.table(feat_vec)


    # add connection to multiple_models app
    # predict category based on predicted features
if __name__ == '__main__':
    main()
