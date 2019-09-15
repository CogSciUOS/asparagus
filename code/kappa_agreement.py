from sklearn.metrics import cohen_kappa_score

import pandas as pd


y_true = [2, 0, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]

print(cohen_kappa_score(y_true, y_pred))
