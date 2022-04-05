from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from Gridsearch import gridsearch
from GBNN import GNEGNERegressor
from scipy.io import arff
import pandas as pd
import numpy as np
import warnings
import os


warnings.simplefilter("ignore")

name = "andro.arff"
d = 30
random_state = 1

np.random.seed(random_state)

# Path to clone the mtr datasets
path = 'https://github.com/lefman/mulan-extended/tree/master/datasets'


def dt(path):
    df = arff.loadarff(path)
    df = pd.DataFrame(df[0])
    return df


def df(name, d):
    dt_name = name
    dt_path = os.path.join(path, dt_name)
    df = dt(dt_path)
    X = (df.iloc[:, :d]).values
    y = (df.iloc[:, d:]).values
    return X, y


X, y = df(name, d)


model_gbnn = GNEGNERegressor(total_nn=200,
                             num_nn_step=1,
                             eta=1.0,
                             solver='lbfgs',
                             subsample=0.5,
                             tol=0.0,
                             max_iter=200,
                             random_state=random_state,
                             activation='logistic')


model_nn = MLPRegressor(solver='adam', hidden_layer_sizes=(
    12,), random_state=random_state)


param_grid_nn = {'reg__hidden_layer_sizes': np.array(
    [1, 3, 5, 7, 11, 12, 17, 22, 27, 32, 37,
     42, 47, 52, 60, 70, 80, 90, 100, 150, 200])}


param_grid_gbnn = {'reg__num_nn_step': [1, 2, 3],
                   'reg__subsample': [0.5, 0.75, 1],
                   'reg__eta': [0.1, 0.025, 0.5, 1]
                   }

if __name__ == "__main__":

    gridsearch(X=X,
               y=y,
               model=model_gbnn,
               grid=param_grid_gbnn,
               scoring_functions='r2',
               pipeline=('scaler', StandardScaler()),
               random_state=random_state,
               n_cv_general=3,
               n_cv_intrain=3,
               title=name + '_GBNN_'
               )

    gridsearch(X=X,
               y=y,
               model=model_nn,
               grid=param_grid_nn,
               scoring_functions='r2',
               pipeline=('scaler', StandardScaler()),
               random_state=random_state,
               n_cv_general=3,
               n_cv_intrain=3,
               title=name + '_NN_'
               )
