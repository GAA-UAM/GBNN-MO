from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from Gridsearch import gridsearch
from GBNN import GNEGNERegressor
from Datasets import dataset
import pandas as pd
import numpy as np
import warnings


warnings.simplefilter("ignore")
random_state = 123
np.random.seed(random_state)
np.set_printoptions(precision=4, suppress=True)


name = "andro.arff"
# Importing the dataset
info = pd.read_csv('Datasets\info.txt')
index = info[info['Dataset'].str.contains(name)]
X, y = dataset(name=str(index.iloc[0, 0]),
               d=int(index.iloc[0, 1]))


model_gbnn = GNEGNERegressor(total_nn=200,
                             num_nn_step=1,
                             eta=1.0,
                             solver='lbfgs',
                             subsample=0.5,
                             tol=0.0,
                             max_iter=2,
                             random_state=random_state,
                             activation='logistic')


model_nn = MLPRegressor(solver='adam', hidden_layer_sizes=(
    12,), random_state=random_state)


param_grid_gbnn = {'reg__num_nn_step': [1, 2, 3],
                   'reg__subsample': [0.5, 0.75, 1],
                   'reg__eta': [0.1, 0.025, 0.5, 1]
                   }


param_grid_nn = {'reg__hidden_layer_sizes': np.array(
    [1, 3, 5, 7, 11, 12, 17, 22, 27, 32, 37,
     42, 47, 52, 60, 70, 80, 90, 100, 150, 200])}

if __name__ == "__main__":

    # Train the single-output model
    for i in range(y.shape[1]):
        Y = y[:, i]

        gridsearch(X=X,
                   y=Y,
                   model=model_gbnn,
                   grid=param_grid_gbnn,
                   scoring_functions='neg_root_mean_squared_error',
                   pipeline=('scaler', StandardScaler()),
                   random_state=random_state,
                   n_cv_general=3,
                   n_cv_intrain=3,
                   title=name[:-5] + "_GBNN_SO_target_" + str(i),
                   )

        gridsearch(X=X,
                   y=Y,
                   model=model_nn,
                   grid=param_grid_nn,
                   scoring_functions='r2',
                   pipeline=('scaler', StandardScaler()),
                   random_state=random_state,
                   n_cv_general=3,
                   n_cv_intrain=3,
                   title=name[:-5] + '_NN_SO_target_'
                   )

    # Train the multi-output model
    gridsearch(X=X,
               y=y,
               model=model_gbnn,
               grid=param_grid_gbnn,
               scoring_functions='r2',
               pipeline=('scaler', StandardScaler()),
               random_state=random_state,
               n_cv_general=3,
               n_cv_intrain=3,
               title=name + '_GBNN_MO_'
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
               title=name + '_NN_MO_'
               )
