# %%
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
# path = r"D:\Academic\Ph.D\Programming\DataBase\PhD Thesis\Regression\mtr_datasets"


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

# name and features for different datasets

# | name | d | missing values |
# andro.arff, 30, False
# atp1d.arff, 411, False
# atp7d.arff, 411, False
# edm.arff, 16, False
# enb.arff, 8
# jura.arff, 15
# rf1.arff, 64


name = "andro.arff"
d = 30

if __name__ == "__main__":
    X, y = df(name, d)[0], df(name, d)[1]


# %% rf1
dt_name = "enb.arff"
dt_path = os.path.join(path, dt_name)
df = dt(dt_path)
X = (df.iloc[:, :8])
y = (df.iloc[:, 8:])
#%%
