# Single-Label CNN

Here you can find the single-label CNN, called 'Asparanet'. It corresponds to the neural network described in the report chapter 4.1.2.   
In the folder `asparanet` is the main file for defining and training the models (`asparanet.py`) together with the code to run it on GRID (`run_asparanet.sge`), the 13 trained models (`trained-models`) and their log files (`trained-models-logs`). In the folder `analyze` are additional function to generate a ROC curve, to count the feature presence in the manually labeled image data, and to let both run in the GRID.   
   
-   asparanet
    -   asparanet.py
    -   run_asparanet.sge
    -   trained-models
    -   trained-models-logs
-   analyze
    -   count_features.py
    -   make_roc_curve.py
    -   run_count_features.sge
    -   run_make_roc_curve.sge
