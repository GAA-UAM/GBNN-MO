import os
import warnings
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.impute import SimpleImputer

warnings.simplefilter("ignore")


# path = 'https://github.com/lefman/mulan-extended/tree/master/datasets'
path = r'D:\Academic\Ph.D\Programming\DataBase\PhD Thesis\Regression\mtr_datasets'

# A method to return the dataset input and targets
# name is the string value of the dataset name
# d is the number of features


def dataset(name, d):
    def dt(path):
        df = arff.loadarff(path)
        df = pd.DataFrame(df[0])
        return df

    dt_name = name
    dt_path = os.path.join(path, dt_name)
    df = dt(dt_path)

    if name == "rf1.arff" or "rf2.arff":
        for i in df.columns:
            df[i] = df[i].fillna(0)
    X = (df.iloc[:, :d]).values
    y = (df.iloc[:, d:]).values
    if name == "scpf.arff":
        imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        X = imp.fit_transform(X)
    return X, y
