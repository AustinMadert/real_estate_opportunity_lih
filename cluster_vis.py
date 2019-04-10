#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
np.random.seed(5)

df = pd.read_pickle('boosted_dataset.pkl')


X = df[['dist_to_downtown', 'price_per_sqft', 'bedrooms']].values

estimators = [('k_means_4', KMeans(n_clusters=4)),
              ('k_means_3', KMeans(n_clusters=3)),
              ('k_means_2', KMeans(n_clusters=2))]

fignum = 1
titles = ['4 clusters', '3 clusters', '2 clusters']
for name, est in estimators:
    fig = plt.figure(fignum, figsize=(12, 12))
    ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=48, azim=134)
    est.fit(X)
    labels = est.labels_

    ax.scatter(X[:, 0], X[:, 1], X[:, 2],
               c=labels.astype(np.float), edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Distance')
    ax.set_ylabel('Price/Sqft')
    ax.set_zlabel('Population')
    ax.set_title(titles[fignum - 1])
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








