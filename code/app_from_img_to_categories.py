from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow.keras as keras
from skimage.io import imread

from labelCNN.training import load_df


IMAGE_COLUMNS = ['image_a', 'image_b', 'image_c']
AUTO_COLUMNS = ['auto_violet', 'auto_length', 'auto_width', 'auto_bended']
LABEL_COLUMNS = ['is_hollow', 'has_blume', 'has_rost_head',
                 'has_rost_body', 'is_bended', 'is_violet']


def load_model():
    pass


def select_data():
    pass


def predict_features(model, input_data):
    pass


def evaluate(true_labels, predicted_labels):
    pass


def main():
    st.write()
    """# Asparagus label prediction"""
    # load the model

    model = load_model()

    # if model is None:
    #    return

    # show information about the model

    # inspect / select data

    # use the loaded model to make predicitons

    # evaluate if prediction was correct

    # add connection to multiple_models app
    # predict category based on predicted features


if __name__ == '__main__':
    main()
