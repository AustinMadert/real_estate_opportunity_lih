#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
np.random.seed(5)

class cluster_visualizer():

    def __init__(self, 
        df_path='boosted_dataset.pkl', 
        estimators=[('k_means_3', KMeans(n_clusters=3))],
        titles=['3 clusters'],
        x_label='Latitude',
        y_label='Longitude',
        z_label='Price'):

        self.df = pd.read_pickle(df_path)
        self.estimators = estimators
        self.titles = titles
        self.x_label='Latitude'
        self.y_label='Longitude'
        self.z_label='Price'


X = df[['dist_to_downtown', 'price_per_sqft', 'bedrooms']].values

    def three_dimension_plot(self, rect=[0, 0, 1, 1], elev=48, azim=134):

        fignum = 1

        for name, est in self.estimators:
            fig = plt.figure(fignum, figsize=(12, 12))
            ax = Axes3D(fig, rect=rect, elev=elev, azim=azim)
            est.fit(X)
            labels = est.labels_

            ax.scatter(X[:, 0], X[:, 1], X[:, 2],
                    c=labels.astype(np.float), edgecolor='k')

            ax.w_xaxis.set_ticklabels([])
            ax.w_yaxis.set_ticklabels([])
            ax.w_zaxis.set_ticklabels([])
            ax.set_xlabel(self.x_label)
            ax.set_ylabel(self.y_label)
            ax.set_zlabel(self.z_label)
            ax.set_title(self.titles[fignum - 1])
            ax.dist = 12
            fignum += 1

        fig.show()



X = df[['min_dist_to_transport', 'price_per_bed', 'price_per_sqft']].values

maximum = X[:,0].argmax()
X[maximum, :]

X = np.delete(X, (maximum), axis=0)

estimators = [('k_means_4', KMeans(n_clusters=4)),
              ('k_means_3', KMeans(n_clusters=3)),
              ('k_means_2', KMeans(n_clusters=2))]

fignum = 1
titles = ['4 clusters', '3 clusters', '2 clusters']
for name, est in estimators:
    fig = plt.figure(fignum, figsize=(12, 12))
    ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=32, azim=150)
    est.fit(X)
    labels = est.labels_

    ax.scatter(X[:, 0], X[:, 1], X[:, 2],
               c=labels.astype(np.float), edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Distance')
    ax.set_ylabel('Price/Bed')
    ax.set_zlabel('Price/Sqft')
    ax.set_title(titles[fignum - 1])
    ax.dist = 12
    fignum += 1








