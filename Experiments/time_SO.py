import warnings
import numpy as np
import pandas as pd
from Datasets import dataset
from time import process_time
from GBNN import GNEGNERegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

random_state = 1
np.random.seed(random_state)

warnings.simplefilter("ignore")


class time_m():
    def __init__(self, n=1000, dt_names=["scm1d.arff",
                                         "scm20d.arff",
                                         "rf1.arff",
                                         "rf2.arff",
                                         "edm.arff",
                                         "enb.arff"],
                 dataset_info='Datasets\info.txt'):
        self.n = n
        self.dt_names = dt_names
        self.datasetf_info = dataset_info

    def input(self, i):
        names = pd.read_csv(self.datasetf_info)
        dt_index = names[names['Dataset'].str.contains(self.dt_names[i])]
        X, y = dataset(dt_index.iloc[0, 0], int(dt_index.iloc[0, 1]))
        return X, y

    def training(self, n=1000):
        results = {"training_time_GBNN": [], "pred_time_GBNN": [],
                   "training_time_NN": [], "pred_time_NN": []}

        for i in range(len(self.dt_names)):
            X, y = self.input(i)

            x_train, x_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=random_state)

            training_time = []
            prediction_time = []

            for k in range(y_train.shape[1]):
                Y_train = y_train[:, k]

                gbnn_mo = GNEGNERegressor(total_nn=200,
                                          num_nn_step=1,
                                          eta=1.0,
                                          solver='lbfgs',
                                          subsample=0.5,
                                          tol=0.0,
                                          max_iter=200,
                                          random_state=random_state,
                                          activation='logistic')

                t0 = process_time()
                gbnn_mo.fit(x_train, Y_train)
                training_time.append((process_time() - t0))

                new_model = gbnn_mo.to_NN()

                t0 = process_time()
                for _ in range(n):
                    new_model.predict(x_test)
                prediction_time.append(
                    (process_time() - t0) / (n * x_test.shape[0]))

            training_time = np.sum(training_time)
            prediction_time = np.sum(prediction_time)

            results["training_time_GBNN"].append(training_time)
            results["pred_time_GBNN"].append(prediction_time)

            training_time = []
            prediction_time = []

            for k in range(y_train.shape[1]):
                Y_train = y_train[:, k]

                mlp = MLPRegressor(hidden_layer_sizes=(200,),
                                   max_iter=200,
                                   tol=0.0,
                                   random_state=random_state,
                                   solver='adam',
                                   activation='logistic')

                t0 = process_time()
                mlp.fit(x_train, Y_train)
                training_time.append((process_time() - t0))

                t0 = process_time()
                for i in range(n):
                    mlp.predict(x_test)
                prediction_time.append(
                    (process_time() - t0) / (n * x_test.shape[0]))

            training_time = np.sum(training_time)
            prediction_time = np.sum(prediction_time)

            results["training_time_NN"].append(training_time)
            results["pred_time_NN"].append(prediction_time)

            print("*", end='')

        pd.DataFrame(results, index=self.dt_names).to_csv("time_SO.csv")


if __name__ == "__main__":
    time_m().training()
