#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from cluster_vis import cluster_visualizer
from sklearn.model_selection import train_test_split
from geoplotting import create_geoplot
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def main():
    df = pd.read_pickle('boosted_dataset.pkl')

    # X_train, X_test = train_test_split(df, test_size=0.1)
    df1 = df.iloc[:1000, :]

    x = df1['lat'].values
    y = df1['lon'].values
    z = df1['price'].values

    x=np.unique(x)
    y=np.unique(y)
    X,Y = np.meshgrid(x,y)

    Z=z[:-1].reshape(-1,len(x))

    plt.pcolormesh(X,Y,Z)

    plt.show()

    return None

if __name__ == '__main__':
    main()






