import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_correlation_matrix(data, columns=None):
    try:
        data = data[columns]
    except:
        pass

    f, ax = plt.subplots(figsize=(10, 8))
    corr = data.corr()
    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
                square=True, ax=ax)
    plt.show()

    return 0

def plot_scatter_matrix(data, columns=None, groupby=None):
    sns.set(style="ticks")

    try:
        data = data[columns]
    except:
        pass

    if groupby is not None:
        sns.pairplot(data,hue=groupby)
    else:
        sns.pairplot(data)
    plt.show()

    return 0