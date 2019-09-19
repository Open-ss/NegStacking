# NegStacking
This is the repository containing the source code for research, namely NegStacking: drug-target interaction prediction based on ensemble learning and logistic regression. The identification of drug-target interactions (DTIs) is an important part of drug research, and many methods proposed to predict potential DTIs based on machine learning treat it as a binary classification problem. However, the number of known interacting drug-target pairs (positive samples) is far less than that of non-interacting pairs (negative samples). Most methods do not utilize these large numbers of negative samples sufficiently, which limits the effect of the prediction method. In order to make full use of negative samples to improve the prediction effect, we proposed a stacked framework named NegStacking. First, it uses sampling without replacement to obtain multiple, completely different, negative sample sets. Then, each weak learner is trained with a different negative sample set and the same positive sample set, and the output of these weak learners is used to train the logistic regression (LR). Finally, the trained model is used to predict new samples. In addition, we used feature subspacing and hyperparameter perturbation to increase ensemble diversity with the purpose of enhancing prediction performance.


## Pre requisites
* numpy
* sklearn
* tensorflow
* python 3.6

## How it works
The data folder contains the data set used in the experiment, including multiple cell line data. In this sudy, we used drug perturbation and gene knockout transcriptome data from the following seven cell lines: PC3, VCAP, A375, HEPG2, HCC515, HA1E, and A549, containing enough data for training.

used_models.py is the model source file.

utils.py is is the tool methods source file.

sensitive_*.py are the model parameter sensitivity analysis source file.

compare.py is the source code file of the model compared with the conventional machine learning methods.
