from sklearn.metrics import cohen_kappa_score
import sys

import pandas as pd
import numpy as np


annotations_1 = pd.read_csv("../annotations/annotator_1.csv", delimiter=";")
annotations_2 = pd.read_csv("../annotations/annotator_2.csv", delimiter=";")

if np.any(annotations_1.isna()) or np.any(annotations_2.isna()):
    print("There are missing values in the csv file. Fix it and come back here!")
    sys.exit(1)

for column in annotations_1.columns[1:]:
    print("For the category", column, "the kappa is:", cohen_kappa_score(
        annotations_1[column], annotations_2[column], labels=[1.0, 0.0]))
