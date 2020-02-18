import streamlit as st

import numpy as np
import pandas as pd
import time

import train_model as tm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


from os import listdir
from os.path import isfile, join

"""
# Visualizing the results of multiple models
Here's our first attempt at displaying the data
"""


@st.cache
def load_data():
    return tm.load_data("../../annotations")


raw_data, dummy_data = load_data()
labels = [col for col in dummy_data if col.startswith('Class')]

if st.checkbox('Show training dataframe'):
    dummy_data

"We want to learn the following labels: ", labels
"There are", len(labels), "labels."

# draw histogram to see how classes are distributed
# TODO, move this to multple models
st.bar_chart(raw_data['Class'].value_counts())

x = dummy_data.iloc[:, :-len(labels)].values
# set Label as y
y = dummy_data[labels].values

# make a train and test split
# 75% train; 25% test
# TODO maybe add a slider here?  Well that is cheating isn't it?
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)

"After we perform the train/test-split, we have", len(
    x_train), "training samples"
"After we perform the train/test-split, we have", len(x_test), "test samples"


"""
# Choose the model that you want to train
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


'## Fitting model'

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

"""## Results

### Classification report"""

st.write(pd.DataFrame(classification_report(y_test.argmax(axis=1),
                                            y_pred.argmax(axis=1), target_names=labels, output_dict=True)).transpose())


"""### Examples
Looking at the first prediction"""

st.write("feature vector", str(y_test[0]),
         " has prediction ", str(y_pred[0].round(3)))

st.write("Use argmax to get labels: sample label", y_test.argmax(axis=1)[
         0], " has prediction sample label", y_pred.argmax(axis=1)[0])


# visualize predictions with images


# show corresponding annotations to image


# show corresponding prediction to image if it is in test split
