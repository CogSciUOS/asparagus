KAPPA OUTPUT 20.11.2019

!!(be aware, the name_kappa_categorie.csv files in the annotation folder are additionally EDITED (rows with empty values deleted), so that the kappa_agreement.py script would run. The original files are in our asparagus project folder in the univestity network, in the folder: Images/labled/kappa_images/results)!!

1A ANNA

/home/student/m/mspaniol/.local/share/virtualenvs/malin-dvmxCYvd/lib/python3.7/site-packages/sklearn/metrics/classification.py:576: RuntimeWarning: invalid value encountered in true_divide
  k = np.sum(w_mat * confusion) / np.sum(w_mat * expected)
For the category is_hollow, the kappa is: 0.0 ❌
For the category has_blume, the kappa is: nan ❌
For the category has_rost_head, the kappa is: 0.3801652892561983 ❌
For the category has_rost_body, the kappa is: 0.0 ❌
For the category is_bended, the kappa is: 0.0 ❌
For the category is_violet, the kappa is: nan ❌
/home/student/m/mspaniol/.local/share/virtualenvs/malin-dvmxCYvd/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
/home/student/m/mspaniol/.local/share/virtualenvs/malin-dvmxCYvd/lib/python3.7/site-packages/sklearn/metrics/classification.py:1439: UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples.
  'recall', 'true', average, warn_for)
=====is_hollow======
               precision    recall  f1-score   support

           0       0.99      1.00      0.99        99
           1       0.00      0.00      0.00         1

    accuracy                           0.99       100
   macro avg       0.49      0.50      0.50       100
weighted avg       0.98      0.99      0.99       100

=====has_blume======
               precision    recall  f1-score   support

           0       1.00      1.00      1.00       100

    accuracy                           1.00       100
   macro avg       1.00      1.00      1.00       100
weighted avg       1.00      1.00      1.00       100

===has_rost_head====
               precision    recall  f1-score   support

           0       0.94      1.00      0.97        92
           1       1.00      0.25      0.40         8

    accuracy                           0.94       100
   macro avg       0.97      0.62      0.68       100
weighted avg       0.94      0.94      0.92       100

===has_rost_body====
               precision    recall  f1-score   support

           0       0.91      1.00      0.95        91
           1       0.00      0.00      0.00         9

    accuracy                           0.91       100
   macro avg       0.46      0.50      0.48       100
weighted avg       0.83      0.91      0.87       100

=====is_bended======
               precision    recall  f1-score   support

           0       1.00      0.98      0.99       100
           1       0.00      0.00      0.00         0

    accuracy                           0.98       100
   macro avg       0.50      0.49      0.49       100
weighted avg       1.00      0.98      0.99       100

=====is_violet======
               precision    recall  f1-score   support

           0       1.00      1.00      1.00       100

    accuracy                           1.00       100
   macro avg       1.00      1.00      1.00       100
weighted avg       1.00      1.00      1.00       100



1A BONA  - throws error!
python kappa_agreement.py ../annotations/malin_kappa_1A_Bona.csv ../annotations/maren_kappa_1A_Bona.csv agreement_malin_maren_1a_bona.csv

error fixed:
Scores above .8 are generally considered good agreement; zero or lower means no agreement (practically random labels)!

/Users/Malin/anaconda3/lib/python3.6/site-packages/sklearn/metrics/classification.py:576: RuntimeWarning: invalid value encountered in true_divide
  k = np.sum(w_mat * confusion) / np.sum(w_mat * expected)
For the category is_hollow, the kappa is: nan ❌
For the category has_blume, the kappa is: 0.0 ❌
For the category has_rost_head, the kappa is: 0.0 ❌
For the category has_rost_body, the kappa is: 0.0 ❌
For the category is_bended, the kappa is: 0.38524590163934425 ❌
For the category is_violet, the kappa is: nan ❌
/Users/Malin/anaconda3/lib/python3.6/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
/Users/Malin/anaconda3/lib/python3.6/site-packages/sklearn/metrics/classification.py:1439: UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples.
  'recall', 'true', average, warn_for)
=====is_hollow======
               precision    recall  f1-score   support

           0       1.00      1.00      1.00       100

    accuracy                           1.00       100
   macro avg       1.00      1.00      1.00       100
weighted avg       1.00      1.00      1.00       100

=====has_blume======
               precision    recall  f1-score   support

           0       0.99      1.00      0.99        99
           1       0.00      0.00      0.00         1

    accuracy                           0.99       100
   macro avg       0.49      0.50      0.50       100
weighted avg       0.98      0.99      0.99       100

===has_rost_head====
               precision    recall  f1-score   support

           0       1.00      0.99      0.99       100
           1       0.00      0.00      0.00         0

    accuracy                           0.99       100
   macro avg       0.50      0.49      0.50       100
weighted avg       1.00      0.99      0.99       100

===has_rost_body====
               precision    recall  f1-score   support

           0       1.00      0.98      0.99       100
           1       0.00      0.00      0.00         0

    accuracy                           0.98       100
   macro avg       0.50      0.49      0.49       100
weighted avg       1.00      0.98      0.99       100

=====is_bended======
               precision    recall  f1-score   support

           0       0.99      0.98      0.98        98
           1       0.33      0.50      0.40         2

    accuracy                           0.97       100
   macro avg       0.66      0.74      0.69       100
weighted avg       0.98      0.97      0.97       100

=====is_violet======
               precision    recall  f1-score   support

           0       1.00      1.00      1.00       100

    accuracy                           1.00       100
   macro avg       1.00      1.00      1.00       100
weighted avg       1.00      1.00      1.00       100

(base) 83adbb96:code Malin$



1A CLARA - throws error: missing values
 python kappa_agreement.py ../annotations/richard_kappa_1A_Clara.csv ../annotations/maren_kappa_1A_Clara.csv agreement_richard_maren_1a_clara.csv

(base) 83adbb96:code Malin$ python kappa_agreement.py ../annotations/richard_kappa_1A_Clara.csv ../annotations/maren_kappa_1A_Clara.csv agreement_richard_maren_1a_clara.csv
Scores above .8 are generally considered good agreement; zero or lower means no agreement (practically random labels)!

For the category is_hollow, the kappa is: 0.0 ❌
For the category has_blume, the kappa is: 0.0 ❌
For the category has_rost_head, the kappa is: 0.49230769230769234 ❌
For the category has_rost_body, the kappa is: 0.646112600536193 ❌
For the category is_bended, the kappa is: 0.3373493975903614 ❌
For the category is_violet, the kappa is: 0.0 ❌
/Users/Malin/anaconda3/lib/python3.6/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
=====is_hollow======
               precision    recall  f1-score   support

           0       0.98      1.00      0.99        97
           1       0.00      0.00      0.00         2

    accuracy                           0.98        99
   macro avg       0.49      0.50      0.49        99
weighted avg       0.96      0.98      0.97        99

=====has_blume======
               precision    recall  f1-score   support

           0       0.97      1.00      0.98        96
           1       0.00      0.00      0.00         3

    accuracy                           0.97        99
   macro avg       0.48      0.50      0.49        99
weighted avg       0.94      0.97      0.95        99

===has_rost_head====
               precision    recall  f1-score   support

           0       0.98      1.00      0.99        96
           1       1.00      0.33      0.50         3

    accuracy                           0.98        99
   macro avg       0.99      0.67      0.74        99
weighted avg       0.98      0.98      0.97        99

===has_rost_body====
               precision    recall  f1-score   support

           0       0.94      0.96      0.95        85
           1       0.75      0.64      0.69        14

    accuracy                           0.92        99
   macro avg       0.85      0.80      0.82        99
weighted avg       0.92      0.92      0.92        99

=====is_bended======
               precision    recall  f1-score   support

           0       0.82      0.95      0.88        75
           1       0.67      0.33      0.44        24

    accuracy                           0.80        99
   macro avg       0.74      0.64      0.66        99
weighted avg       0.78      0.80      0.77        99

=====is_violet======
               precision    recall  f1-score   support

           0       0.98      1.00      0.99        97
           1       0.00      0.00      0.00         2

    accuracy                           0.98        99
   macro avg       0.49      0.50      0.49        99
weighted avg       0.96      0.98      0.97        99


1A KRUMME

(malin) (base) mspaniol@gate:/net/projects/scratch/summer/valid_until_31_January_2020/asparagus/malin/asparagus/code$ python kappa_agreement.py ../annotations/sophia_kappa_1A_Krumme.csv ../annotations/michael_kappa_1A_Krumme.csv agreement_sophia_michael_1a_krumme.csv
Scores above .8 are generally considered good agreement; zero or lower means no agreement (practically random labels)!

For the category is_hollow, the kappa is: -0.010101010101010166 ❌
For the category has_blume, the kappa is: 0.3690851735015773 ❌
For the category has_rost_head, the kappa is: 0.0 ❌
For the category has_rost_body, the kappa is: 0.3202416918429002 ❌
For the category is_bended, the kappa is: 0.5950554134697357 ❌
For the category is_violet, the kappa is: 0.0 ❌
/home/student/m/mspaniol/.local/share/virtualenvs/malin-dvmxCYvd/lib/python3.7/site-packages/sklearn/metrics/classification.py:1439: UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples.
  'recall', 'true', average, warn_for)
/home/student/m/mspaniol/.local/share/virtualenvs/malin-dvmxCYvd/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
=====is_hollow======
               precision    recall  f1-score   support

           0       0.99      0.99      0.99        99
           1       0.00      0.00      0.00         1

    accuracy                           0.98       100
   macro avg       0.49      0.49      0.49       100
weighted avg       0.98      0.98      0.98       100

=====has_blume======
               precision    recall  f1-score   support

           0       0.62      0.88      0.72        48
           1       0.81      0.50      0.62        52

    accuracy                           0.68       100
   macro avg       0.72      0.69      0.67       100
weighted avg       0.72      0.68      0.67       100

===has_rost_head====
               precision    recall  f1-score   support

           0       1.00      0.99      0.99       100
           1       0.00      0.00      0.00         0

    accuracy                           0.99       100
   macro avg       0.50      0.49      0.50       100
weighted avg       1.00      0.99      0.99       100

===has_rost_body====
               precision    recall  f1-score   support

           0       0.83      0.97      0.89        78
           1       0.75      0.27      0.40        22

    accuracy                           0.82       100
   macro avg       0.79      0.62      0.65       100
weighted avg       0.81      0.82      0.79       100

=====is_bended======
               precision    recall  f1-score   support

           0       0.93      0.60      0.73        43
           1       0.76      0.96      0.85        57

    accuracy                           0.81       100
   macro avg       0.85      0.78      0.79       100
weighted avg       0.83      0.81      0.80       100

=====is_violet======
               precision    recall  f1-score   support

           0       0.99      1.00      0.99        99
           1       0.00      0.00      0.00         1

    accuracy                           0.99       100
   macro avg       0.49      0.50      0.50       100
weighted avg       0.98      0.99      0.99       100

1A Violett

python kappa_agreement.py ../annotations/josefine_kappa_1A_Violett.csv ../annotations/malin_kappa_1A_Violett.csv agreement_josefine_malin_1a_violet.csv

: RuntimeWarning: invalid value encountered in true_divide
  k = np.sum(w_mat * confusion) / np.sum(w_mat * expected)
For the category is_hollow, the kappa is: nan ❌
For the category has_blume, the kappa is: 0.0 ❌
For the category has_rost_head, the kappa is: 0.5923913043478262 ❌
For the category has_rost_body, the kappa is: 0.47916666666666663 ❌
For the category is_bended, the kappa is: 0.0 ❌
For the category is_violet, the kappa is: 0.467275494672755 ❌
/home/student/m/mspaniol/.local/share/virtualenvs/malin-dvmxCYvd/lib/python3.7/site-packages/sklearn/metrics/classification.py:1439: UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples.
  'recall', 'true', average, warn_for)
=====is_hollow======
               precision    recall  f1-score   support

           0       1.00      1.00      1.00       100

    accuracy                           1.00       100
   macro avg       1.00      1.00      1.00       100
weighted avg       1.00      1.00      1.00       100

=====has_blume======
               precision    recall  f1-score   support

           0       1.00      0.83      0.91       100
           1       0.00      0.00      0.00         0

    accuracy                           0.83       100
   macro avg       0.50      0.41      0.45       100
weighted avg       1.00      0.83      0.91       100

===has_rost_head====
               precision    recall  f1-score   support

           0       0.97      0.97      0.97        92
           1       0.62      0.62      0.62         8

    accuracy                           0.94       100
   macro avg       0.80      0.80      0.80       100
weighted avg       0.94      0.94      0.94       100

===has_rost_body====
               precision    recall  f1-score   support

           0       0.98      0.98      0.98        96
           1       0.50      0.50      0.50         4

    accuracy                           0.96       100
   macro avg       0.74      0.74      0.74       100
weighted avg       0.96      0.96      0.96       100

=====is_bended======
               precision    recall  f1-score   support

           0       1.00      0.88      0.94       100
           1       0.00      0.00      0.00         0

    accuracy                           0.88       100
   macro avg       0.50      0.44      0.47       100
weighted avg       1.00      0.88      0.94       100

=====is_violet======
               precision    recall  f1-score   support

           0       0.55      0.94      0.70        34
           1       0.95      0.61      0.74        66

    accuracy                           0.72       100
   macro avg       0.75      0.77      0.72       100
weighted avg       0.82      0.72      0.73       100


2A - throws error missing values!
 python kappa_agreement.py ../annotations/maren_kappa_2A.csv ../annotations/malin_kappa_2A.csv agreement_maren_malin_2a.csv

For the category is_hollow, the kappa is: 0.15077989601386488 ❌
For the category has_blume, the kappa is: 0.05046427129592257 ❌
For the category has_rost_head, the kappa is: 0.4235294117647058 ❌
For the category has_rost_body, the kappa is: 0.44035532994923854 ❌
For the category is_bended, the kappa is: 0.42503259452412 ❌
For the category is_violet, the kappa is: 0.6486714466469844 ❌
=====is_hollow======
               precision    recall  f1-score   support

           0       0.90      1.00      0.95        87
           1       1.00      0.09      0.17        11

    accuracy                           0.90        98
   macro avg       0.95      0.55      0.56        98
weighted avg       0.91      0.90      0.86        98

=====has_blume======
               precision    recall  f1-score   support

           0       0.49      0.91      0.64        47
           1       0.64      0.14      0.23        51

    accuracy                           0.51        98
   macro avg       0.57      0.53      0.43        98
weighted avg       0.57      0.51      0.43        98

===has_rost_head====
               precision    recall  f1-score   support

           0       0.97      0.96      0.96        92
           1       0.43      0.50      0.46         6

    accuracy                           0.93        98
   macro avg       0.70      0.73      0.71        98
weighted avg       0.93      0.93      0.93        98

===has_rost_body====
               precision    recall  f1-score   support

           0       0.93      0.84      0.88        82
           1       0.46      0.69      0.55        16

    accuracy                           0.82        98
   macro avg       0.70      0.76      0.72        98
weighted avg       0.86      0.82      0.83        98

=====is_bended======
               precision    recall  f1-score   support

           0       0.79      0.52      0.63        44
           1       0.70      0.89      0.78        54

    accuracy                           0.72        98
   macro avg       0.74      0.71      0.71        98
weighted avg       0.74      0.72      0.71        98

=====is_violet======
               precision    recall  f1-score   support

           0       0.81      0.80      0.80        44
           1       0.84      0.85      0.84        54

    accuracy                           0.83        98
   macro avg       0.83      0.82      0.82        98
weighted avg       0.83      0.83      0.83        98

2B - throws error missing values!
python kappa_agreement.py ../annotations/richard_kappa_2B.csv ../annotations/josefine_kappa_2B.csv agreement_richard_josefine_2b.csv

error fixed


(base) 83adbb96:code Malin$ python kappa_agreement.py ../annotations/richard_kappa_2B.csv ../annotations/josefine_kappa_2B.csv agreement_richard_josefine_2b.csv
Scores above .8 are generally considered good agreement; zero or lower means no agreement (practically random labels)!

For the category is_hollow, the kappa is: 0.0 ❌
For the category has_blume, the kappa is: 0.27899159663865547 ❌
For the category has_rost_head, the kappa is: 0.5560538116591929 ❌
For the category has_rost_body, the kappa is: 0.3676366217175301 ❌
For the category is_bended, the kappa is: 0.5180952380952382 ❌
For the category is_violet, the kappa is: 0.6024096385542168 ❌
/Users/Malin/anaconda3/lib/python3.6/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
=====is_hollow======
               precision    recall  f1-score   support

           0       0.97      1.00      0.98        96
           1       0.00      0.00      0.00         3

    accuracy                           0.97        99
   macro avg       0.48      0.50      0.49        99
weighted avg       0.94      0.97      0.95        99

=====has_blume======
               precision    recall  f1-score   support

           0       0.86      1.00      0.93        83
           1       1.00      0.19      0.32        16

    accuracy                           0.87        99
   macro avg       0.93      0.59      0.62        99
weighted avg       0.89      0.87      0.83        99

===has_rost_head====
               precision    recall  f1-score   support

           0       0.99      0.98      0.98        96
           1       0.50      0.67      0.57         3

    accuracy                           0.97        99
   macro avg       0.74      0.82      0.78        99
weighted avg       0.97      0.97      0.97        99

===has_rost_body====
               precision    recall  f1-score   support

           0       0.80      1.00      0.89        74
           1       1.00      0.28      0.44        25

    accuracy                           0.82        99
   macro avg       0.90      0.64      0.66        99
weighted avg       0.85      0.82      0.78        99

=====is_bended======
               precision    recall  f1-score   support

           0       0.65      0.78      0.71        36
           1       0.86      0.76      0.81        63

    accuracy                           0.77        99
   macro avg       0.75      0.77      0.76        99
weighted avg       0.78      0.77      0.77        99

=====is_violet======
               precision    recall  f1-score   support

           0       1.00      0.86      0.93        87
           1       0.50      1.00      0.67        12

    accuracy                           0.88        99
   macro avg       0.75      0.93      0.80        99
weighted avg       0.94      0.88      0.89        99


BLUME - throws error
python kappa_agreement.py ../annotations/malin_kappa_blume.csv ../annotations/sophia_kappa_blume.csv agreement_malin_sophia_blume.csv
Traceback (most recent call last):


For the category is_hollow, the kappa is: -0.01379310344827589 ❌
For the category has_blume, the kappa is: 0.3502762430939227 ❌
For the category has_rost_head, the kappa is: 0.0 ❌
For the category has_rost_body, the kappa is: -0.03157894736842115 ❌
For the category is_bended, the kappa is: 0.6127382612738261 ❌
For the category is_violet, the kappa is: 0.42622950819672123 ❌
/Users/Malin/anaconda3/lib/python3.6/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
=====is_hollow======
               precision    recall  f1-score   support

           0       0.98      0.99      0.98        96
           1       0.00      0.00      0.00         2

    accuracy                           0.97        98
   macro avg       0.49      0.49      0.49        98
weighted avg       0.96      0.97      0.96        98

=====has_blume======
               precision    recall  f1-score   support

           0       0.27      0.80      0.40         5
           1       0.99      0.88      0.93        93

    accuracy                           0.88        98
   macro avg       0.63      0.84      0.67        98
weighted avg       0.95      0.88      0.90        98

===has_rost_head====
               precision    recall  f1-score   support

           0       0.99      1.00      0.99        97
           1       0.00      0.00      0.00         1

    accuracy                           0.99        98
   macro avg       0.49      0.50      0.50        98
weighted avg       0.98      0.99      0.98        98

===has_rost_body====
               precision    recall  f1-score   support

           0       0.97      0.97      0.97        95
           1       0.00      0.00      0.00         3

    accuracy                           0.94        98
   macro avg       0.48      0.48      0.48        98
weighted avg       0.94      0.94      0.94        98

=====is_bended======
               precision    recall  f1-score   support

           0       0.98      0.78      0.87        74
           1       0.59      0.96      0.73        24

    accuracy                           0.83        98
   macro avg       0.79      0.87      0.80        98
weighted avg       0.89      0.83      0.84        98

=====is_violet======
               precision    recall  f1-score   support

           0       1.00      0.95      0.97        96
           1       0.29      1.00      0.44         2

    accuracy                           0.95        98
   macro avg       0.64      0.97      0.71        98
weighted avg       0.99      0.95      0.96        98

DICKE
python kappa_agreement.py ../annotations/malin_kappa_Dicke.csv ../annotations/michael_kappa_Dicke.csv agreement_malin_michael_dicke.csv
Scores above .8 are generally considered good agreement; zero or lower means no agreement (practically random labels)!

For the category is_hollow, the kappa is: 0.0 ❌
For the category has_blume, the kappa is: 0.08732999284180376 ❌
For the category has_rost_head, the kappa is: -0.02941176470588247 ❌
For the category has_rost_body, the kappa is: 0.5159453302961275 ❌
For the category is_bended, the kappa is: 0.4215938303341902 ❌
For the category is_violet, the kappa is: 0.42660550458715596 ❌
/home/student/m/mspaniol/.local/share/virtualenvs/malin-dvmxCYvd/lib/python3.7/site-packages/sklearn/metrics/classification.py:1439: UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples.
  'recall', 'true', average, warn_for)
=====is_hollow======
               precision    recall  f1-score   support

           0       1.00      0.99      0.99       100
           1       0.00      0.00      0.00         0

    accuracy                           0.99       100
   macro avg       0.50      0.49      0.50       100
weighted avg       1.00      0.99      0.99       100

=====has_blume======
               precision    recall  f1-score   support

           0       0.98      0.46      0.62        92
           1       0.12      0.88      0.22         8

    accuracy                           0.49       100
   macro avg       0.55      0.67      0.42       100
weighted avg       0.91      0.49      0.59       100

===has_rost_head====
               precision    recall  f1-score   support

           0       0.87      0.90      0.88        87
           1       0.10      0.08      0.09        13

    accuracy                           0.79       100
   macro avg       0.48      0.49      0.48       100
weighted avg       0.77      0.79      0.78       100

===has_rost_body====
               precision    recall  f1-score   support

           0       0.93      0.85      0.89        81
           1       0.54      0.74      0.62        19

    accuracy                           0.83       100
   macro avg       0.74      0.79      0.76       100
weighted avg       0.86      0.83      0.84       100

=====is_bended======
               precision    recall  f1-score   support

           0       0.95      0.96      0.95        91
           1       0.50      0.44      0.47         9

    accuracy                           0.91       100
   macro avg       0.72      0.70      0.71       100
weighted avg       0.91      0.91      0.91       100

=====is_violet======
               precision    recall  f1-score   support

           0       0.95      1.00      0.97        93
           1       1.00      0.29      0.44         7

    accuracy                           0.95       100
   macro avg       0.97      0.64      0.71       100
weighted avg       0.95      0.95      0.94       100

HOHLE - throws error - missing values
python kappa_agreement.py ../annotations/malin_kappa_Hohle.csv ../annotations/sophia_kappa_Hohle.csv agreement_malin_sophia_hohle.csv

For the category is_hollow, the kappa is: 0.0 ❌
For the category has_blume, the kappa is: 0.4809688581314878 ❌
For the category has_rost_head, the kappa is: 0.41229656419529837 ❌
For the category has_rost_body, the kappa is: 0.19354838709677413 ❌
For the category is_bended, the kappa is: 0.7567896230239157 ❌
For the category is_violet, the kappa is: 0.7402597402597402 ❌
/Users/Malin/anaconda3/lib/python3.6/site-packages/sklearn/metrics/classification.py:1439: UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples.
  'recall', 'true', average, warn_for)
=====is_hollow======
               precision    recall  f1-score   support

           0       0.00      0.00      0.00         0
           1       1.00      0.89      0.94       100

    accuracy                           0.89       100
   macro avg       0.50      0.45      0.47       100
weighted avg       1.00      0.89      0.94       100

=====has_blume======
               precision    recall  f1-score   support

           0       0.99      0.88      0.93        92
           1       0.39      0.88      0.54         8

    accuracy                           0.88       100
   macro avg       0.69      0.88      0.73       100
weighted avg       0.94      0.88      0.90       100

===has_rost_head====
               precision    recall  f1-score   support

           0       0.89      0.96      0.93        84
           1       0.67      0.38      0.48        16

    accuracy                           0.87       100
   macro avg       0.78      0.67      0.70       100
weighted avg       0.85      0.87      0.85       100

===has_rost_body====
               precision    recall  f1-score   support

           0       0.43      1.00      0.60        36
           1       1.00      0.25      0.40        64

    accuracy                           0.52       100
   macro avg       0.71      0.62      0.50       100
weighted avg       0.79      0.52      0.47       100

=====is_bended======
               precision    recall  f1-score   support

           0       0.79      0.95      0.86        39
           1       0.96      0.84      0.89        61

    accuracy                           0.88       100
   macro avg       0.87      0.89      0.88       100
weighted avg       0.89      0.88      0.88       100

=====is_violet======
               precision    recall  f1-score   support

           0       1.00      0.98      0.99        97
           1       0.60      1.00      0.75         3

    accuracy                           0.98       100
   macro avg       0.80      0.99      0.87       100
weighted avg       0.99      0.98      0.98       100

KOEPFE - throws error
python kappa_agreement.py ../annotations/josefine_kappa_Koepfe.csv ../annotations/maren_kappa_Koepfe.csv agreement_josefine_maren_koepfe.csv


For the category is_hollow, the kappa is: 0.0 ❌
For the category has_blume, the kappa is: 0.0 ❌
For the category has_rost_head, the kappa is: 0.5209713024282561 ❌
For the category has_rost_body, the kappa is: 0.2530120481927711 ❌
For the category is_bended, the kappa is: 0.12539184952978044 ❌
For the category is_violet, the kappa is: 0.4521975532396919 ❌
/Users/Malin/anaconda3/lib/python3.6/site-packages/sklearn/metrics/classification.py:1439: UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples.
  'recall', 'true', average, warn_for)
=====is_hollow======
               precision    recall  f1-score   support

           0       1.00      0.99      0.99        93
           1       0.00      0.00      0.00         0

    accuracy                           0.99        93
   macro avg       0.50      0.49      0.50        93
weighted avg       1.00      0.99      0.99        93

=====has_blume======
               precision    recall  f1-score   support

           0       1.00      0.77      0.87        93
           1       0.00      0.00      0.00         0

    accuracy                           0.77        93
   macro avg       0.50      0.39      0.44        93
weighted avg       1.00      0.77      0.87        93

===has_rost_head====
               precision    recall  f1-score   support

           0       0.89      0.76      0.82        63
           1       0.62      0.80      0.70        30

    accuracy                           0.77        93
   macro avg       0.75      0.78      0.76        93
weighted avg       0.80      0.77      0.78        93

===has_rost_body====
               precision    recall  f1-score   support

           0       1.00      0.72      0.84        87
           1       0.20      1.00      0.33         6

    accuracy                           0.74        93
   macro avg       0.60      0.86      0.59        93
weighted avg       0.95      0.74      0.81        93

=====is_bended======
               precision    recall  f1-score   support

           0       1.00      0.77      0.87        91
           1       0.09      1.00      0.16         2

    accuracy                           0.77        93
   macro avg       0.54      0.88      0.51        93
weighted avg       0.98      0.77      0.85        93

=====is_violet======
               precision    recall  f1-score   support

           0       0.86      0.99      0.92        74
           1       0.88      0.37      0.52        19

    accuracy                           0.86        93
   macro avg       0.87      0.68      0.72        93
weighted avg       0.86      0.86      0.84        93

ROST

(base) mspaniol@gate:/net/projects/scratch/summer/valid_until_31_January_2020/asparagus/malin/asparagus/code$ python kappa_agreement.py ../annotations/josefine_kappa_Rost.csv ../annotations/maren_kappa_Rost.csv agreement_josefine_maren_rost.csv
Scores above .8 are generally considered good agreement; zero or lower means no agreement (practically random labels)!

For the category is_hollow, the kappa is: 0.0 ❌
For the category has_blume, the kappa is: 0.3356299212598425 ❌
For the category has_rost_head, the kappa is: 0.48263118994826315 ❌
For the category has_rost_body, the kappa is: 0.5641646489104115 ❌
For the category is_bended, the kappa is: 0.6458333333333333 ❌
For the category is_violet, the kappa is: 0.5901639344262295 ❌
/home/student/m/mspaniol/miniconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1439: UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples.
  'recall', 'true', average, warn_for)
=====is_hollow======
               precision    recall  f1-score   support

           0       1.00      0.99      0.99       100
           1       0.00      0.00      0.00         0

    accuracy                           0.99       100
   macro avg       0.50      0.49      0.50       100
weighted avg       1.00      0.99      0.99       100

=====has_blume======
               precision    recall  f1-score   support

           0       0.97      0.71      0.82        86
           1       0.32      0.86      0.47        14

    accuracy                           0.73       100
   macro avg       0.65      0.78      0.64       100
weighted avg       0.88      0.73      0.77       100

===has_rost_head====
               precision    recall  f1-score   support

           0       0.69      0.47      0.56        19
           1       0.89      0.95      0.92        81

    accuracy                           0.86       100
   macro avg       0.79      0.71      0.74       100
weighted avg       0.85      0.86      0.85       100

===has_rost_body====
               precision    recall  f1-score   support

           0       0.90      0.54      0.68        35
           1       0.80      0.97      0.88        65

    accuracy                           0.82       100
   macro avg       0.85      0.76      0.78       100
weighted avg       0.84      0.82      0.81       100

=====is_bended======
               precision    recall  f1-score   support

           0       0.77      0.98      0.86        55
           1       0.97      0.64      0.77        45

    accuracy                           0.83       100
   macro avg       0.87      0.81      0.82       100
weighted avg       0.86      0.83      0.82       100

=====is_violet======
               precision    recall  f1-score   support

           0       0.96      0.99      0.97        92
           1       0.80      0.50      0.62         8

    accuracy                           0.95       100
   macro avg       0.88      0.74      0.79       100
weighted avg       0.95      0.95      0.94       100


SUPPE - throws error missing values
python kappa_agreement.py ../annotations/richard_kappa_Suppe.csv ../annotations/michael_kappa_Suppe.csv agreement_richard_michael_suppe.csv

Scores above .8 are generally considered good agreement; zero or lower means no agreement (practically random labels)!

/Users/Malin/anaconda3/lib/python3.6/site-packages/sklearn/metrics/classification.py:576: RuntimeWarning: invalid value encountered in true_divide
  k = np.sum(w_mat * confusion) / np.sum(w_mat * expected)
For the category is_hollow, the kappa is: 0.0 ❌
For the category has_blume, the kappa is: 0.0 ❌
For the category has_rost_head, the kappa is: nan ❌
For the category has_rost_body, the kappa is: 0.5570776255707762 ❌
For the category is_bended, the kappa is: 0.5178301275124271 ❌
For the category is_violet, the kappa is: 0.6475930971843779 ❌
/Users/Malin/anaconda3/lib/python3.6/site-packages/sklearn/metrics/classification.py:1439: UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples.
  'recall', 'true', average, warn_for)
=====is_hollow======
               precision    recall  f1-score   support

           0       1.00      0.98      0.99        97
           1       0.00      0.00      0.00         0

    accuracy                           0.98        97
   macro avg       0.50      0.49      0.49        97
weighted avg       1.00      0.98      0.99        97

=====has_blume======
               precision    recall  f1-score   support

           0       1.00      0.35      0.52        97
           1       0.00      0.00      0.00         0

    accuracy                           0.35        97
   macro avg       0.50      0.18      0.26        97
weighted avg       1.00      0.35      0.52        97

===has_rost_head====
               precision    recall  f1-score   support

           0       1.00      1.00      1.00        97

    accuracy                           1.00        97
   macro avg       1.00      1.00      1.00        97
weighted avg       1.00      1.00      1.00        97

===has_rost_body====
               precision    recall  f1-score   support

           0       0.93      0.96      0.95        84
           1       0.70      0.54      0.61        13

    accuracy                           0.91        97
   macro avg       0.82      0.75      0.78        97
weighted avg       0.90      0.91      0.90        97

=====is_bended======
               precision    recall  f1-score   support

           0       0.90      0.72      0.80        64
           1       0.61      0.85      0.71        33

    accuracy                           0.76        97
   macro avg       0.76      0.78      0.75        97
weighted avg       0.80      0.76      0.77        97

=====is_violet======
               precision    recall  f1-score   support

           0       0.99      0.92      0.95        87
           1       0.56      0.90      0.69        10

    accuracy                           0.92        97
   macro avg       0.78      0.91      0.82        97
weighted avg       0.94      0.92      0.93        97
