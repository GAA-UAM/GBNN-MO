# Experiments

This package contains the related experiments to the results of the paper. The comparison had done between two proposed GBNNN-Multioutput and GBNN-SingleOutput and neural networks.
We considered more than 18 datasets for these experiments, and you can access the dataset from the corresponding directory.

## model_training

Contains the implementation of the Multi-Output and Single-Output experiments with the GBNN-MO and NN model.
The training process of the proposed method and neural network with the optimization process are included.

It illustrates one dataset as an example. You can expand it for all the used datasets. 

## Time

The related experiments to measuring the time are included in the (time_MO)[time_MO.py] and (time_SO)[time_SO.py] file. The (time_MO)[time_MO.py] and (time_SO)[time_SO.py] return the training and prediction time for both MO and NO approaches. The training time is equal to the needed time for the mode to learn, and the prediction is the average time for predicting one instance of the selected dataset. For the GBNN, the prediction time would be the time of the shallow neural network, as we used the trained Regression Neural Network with the gradient Boosting approach.

In this experiment, we only selected a subset of the dataset, you can change the class input with your modified list to have the time for the other datasets as well.

## Dataset


A method that returns the input and output of different multi-output regression datasets. We considered MTR datasets for the experiments. You only need to set the cloned path of the MTR Datasets. The approaches to deal with the missing values are included in the method. The strategies to deal with the missing values are based on the dataset's [reference](https://doi.org/10.1007/s10994-016-5546-z).

You can use info.txt to extract the dataset. The dataset extraction is illustrated in the examples.
