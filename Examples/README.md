# Example

This package contains the related experiments to the results of the paper. The comparison had done between two proposed GBNNN-Multioutput and GBNN-SingleOutput and neural networks.
We considered more than 18 datasets for these experiments, and you can access the dataset from the corresponding directory.

# Multioutput

This folder contains the implementation of the Multioutput experiments with the GBNN-MO and NN model.
It illustrates one dataset as an example. You can expand it for all the used datasets.

# Singleoutput
This folder contains the implementation of the Multioutput experiments with the GBNN-SO.
It illustrates one dataset as an example. You can expand it for all the used datasets.

# Time
The related experiments to measuring the time are included in the (time_MO)[time_MO.py] and (time_SO)[time_SO.py] file. The (time_MO)[time_MO.py] and (time_SO)[time_SO.py] return the training and prediction time for both MO and NO approaches. The training time is equal to the needed time for the mode to learn, and the prediction is the average time for predicting one instance of the selected dataset. For the GBNN, the prediction time would be the time of the shallow neural network, as we used the trained Regression Neural Network with the gradient Boosting approach.

In this experiment, we only selected a subset of the dataset, you can change the class input with your modified list to have the time for the other datasets as well.

# Dataset

This directory contains the example of using 17 multioutput regression datasets. I considered MTR datasets for the experiments. The path to clone the dataset and the number of features and missing values are included. The approaches to deal with the missing values are also mentioned in the txt file. You can use the info.txt to extract the dataset. The dataset extraction is illustrated in the [time](time.py).

To have the reference of the datasets, please refer to [here](https://doi.org/10.1007/s10994-016-5546-z).