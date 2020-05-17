# Single-Label CNN

Here you can find the single-label CNN, called 'Asparanet'. It is the neural network described in the report chapter 4.1.2.   
In the folder `asparanet` is the main file for defining and training a model (`asparanet.py`) together with the code to run it on GRID (`run_asparanet.sge`). The 13 trained models can also be found here (`trained-models`) and their corresponding log files (`trained-models-logs`). In the folder `analyze` are additional functions to generate a ROC curve, to count the feature presence in the manually labeled image data, and scripts to let both run in the GRID.   
   
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
