import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.io import arff
import matplotlib.pyplot as plt

warnings.simplefilter("ignore")


random_state = 123
np.random.seed(random_state)


path = 'https://github.com/lefman/mulan-extended/tree/master/datasets'


def dt(path):
    df = arff.loadarff(path)
    df = pd.DataFrame(df[0])
    sns.heatmap(df.isnull(), yticklabels=False, cbar=True)
    plt.show()
    return df


# A method to return the dataset input and targets
# name is the string value of the dataset name
# d is the number of features
def df(name, d):
    dt_name = name
    dt_path = os.path.join(path, dt_name)
    df = dt(dt_path)
    X = (df.iloc[:, :d]).values
    y = (df.iloc[:, d:]).values
    return X, y


name = "oes97.arff"
d = 30

if __name__ == "__main__":
    X, y = df(name, d)
