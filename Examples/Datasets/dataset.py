import os
import warnings
import pandas as pd
from scipy.io import arff
warnings.simplefilter("ignore")


# path = 'https://github.com/lefman/mulan-extended/tree/master/datasets'
path = r'D:\Academic\Ph.D\Programming\DataBase\PhD Thesis\Regression\mtr_datasets'


def dataset(name, d):
    def dt(path):
        df = arff.loadarff(path)
        df = pd.DataFrame(df[0])
        return df

    # A method to return the dataset input and targets
    # name is the string value of the dataset name
    # d is the number of features

    dt_name = name
    dt_path = os.path.join(path, dt_name)
    df = dt(dt_path)
    X = (df.iloc[:, :d]).values
    y = (df.iloc[:, d:]).values
    return X, y
