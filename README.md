Source code for identifying important features for predicting pan-cancer tumor samples from mixed tumor and non-tumor samples. Data set will be released publicly after the manuscript is accepted.

This folder contains three python scripts, the `main.py` script invokes functions from the other two scripts. Usage: `python main.py` will execute the pipeline (all parameters were included in the script). Based on the workflow described in the manuscript (Fig. S5), it will train a pan-cancer machine learning model (using the random forest algorithm) with grid research to determine optimal model paramaters, based on training data. It will then output final feature importance. Finally, it will validate top features using test data.
