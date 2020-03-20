""" This is a streamlit app to compare different models
    
    The models take features that were labeled by humans as input
    (for training we have a hand-sortet ground truth)
    and provide a prediction for a class as output

    - inspection of corresponding image is possible
    - inspection of data frame 
    - inspection of confusion matrix

    To run the app:
    streamlit run multiple_models_app.py
"""
import streamlit as st

import numpy as np
import pandas as pd
import time


import train_model as tm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


from io import StringIO
from pathlib import Path
from skimage.io import imread

from os import listdir
from os.path import isfile, join


def main():
    st.markdown(
        """
    # Visualizing the results of multiple models
    Here's our first attempt at displaying the data
    """)

    # loading the data
    @st.cache
    def load_data():
        return tm.load_data("../../annotations")

    raw_data, dummy_data = load_data()
    labels = [col for col in dummy_data if col.startswith('Class')]

    # display the train df
    if st.checkbox('Show training dataframe'):
        dummy_data

    "There are", len(labels), "labels."
    "We want to learn the following labels: "
    # draw histogram to see how classes are distributed
    st.bar_chart(raw_data['Class'].value_counts())

    ######
    # train the model
    x = dummy_data.iloc[:, :-len(labels)].values
    # set Label as y
    y = dummy_data[labels].values

    # make a train and test split: 75% train; 25% test
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)

    "After we perform the train/test-split, we have", len(
        x_train), "training samples"
    "After we perform the train/test-split, we have", len(
        x_test), "test samples"

    """
    ## Choose the model that you want to train
    Right now there is a random forest model and a multilayer preceptron.
    """
    model_folder = "models"
    models = [model[:model.index(".")] for model in listdir(
        model_folder) if isfile(join(model_folder, model))]

    model_option = st.selectbox(
        'Which model do you want to train?',
        models)
    'You selected: ', model_option

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
        st.write('Score is', np.round(score, 3))

    if hasattr(model, 'evaluate'):
        evaluation = model.evaluate(x_test, y_test)
        st.write('Evaluation is', evaluation)

    # get the predictions
    y_pred = model.predict(x_test)

    "Number of samples in predictions:", len(y_pred)
    ######

    """## Results"""
    """ ### Classification report"""
    st.write(pd.DataFrame(classification_report(y_test.argmax(axis=1),
                                                y_pred.argmax(axis=1), target_names=labels, output_dict=True)).transpose())

    "### Confusion Matrix"
    if st.checkbox('Show confusion matrix'):
        conf_matrix, ax = tm.conf_matrix(
            x_train, x_test, y_train, y_test, y_pred, labels, model_option)
        st.pyplot()

    """ ## Inspecting specific asparagus pieces
    ### Which image folder do you want to have a look at?"""
    kappa_img_folder = "kappa_images"
    class_folders = [class_folder for class_folder in listdir(
        kappa_img_folder)]
    class_folders.sort()

    img_option = st.selectbox(
        '',
        class_folders)

    number_img_folder = len(
        [image for image in listdir(kappa_img_folder+"/"+img_option+"/")])
    st.write("number of images in ", img_option,
             "folder is:", number_img_folder)

    "### Select data"

    # image_dir = Path(st.text_input('Image directory', 'images'))
    # if not image_dir.is_dir():
    #    st.error('Please select a valid image directory.')

    def load_sample(row):
        "#### Selected sample"
        st.dataframe(pd.DataFrame(row).transpose())

        st.write('#### Images (a, b, c)')

        path = kappa_img_folder+"/"+img_option+"/"
        st.write(path)
        images = [img for img in listdir(path)]
        images.sort()

        # this is cheated but there are no filename infos from the label app
        # so always take pairs of three
        images_3 = images[sample_idx*3:sample_idx*3+3].copy()
        st.write(images_3)
        images_3_imread = [imread(path+img) for img in images_3]

        # stack and display 3 images
        st.image(np.hstack([np.squeeze(i)
                            for i in images_3_imread]), use_column_width=True)

    # I am missing the filenames here
    # so this is not totally correct and only for visualization
    df = dummy_data
    sample_idx = st.slider('Sample', 0, int(number_img_folder / 3.0), value=0)
    load_sample(df.iloc[sample_idx])

    "### Examples"
    "Looking at the", sample_idx, "th prediction"

    st.write("feature vector", str(y_test[sample_idx]),
             " has prediction ", str(y_pred[sample_idx].round(3)))

    st.write("Use argmax to get labels: sample label", y_test.argmax(axis=1)[
        sample_idx], " has prediction sample label", y_pred.argmax(axis=1)[sample_idx])


if __name__ == "__main__":
    main()
