#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from joblib import dump
np.random.seed(5)

class cluster_visualizer():


    def __init__(self,
        df_path='boosted_dataset.pkl', 
        X3_features=['lat', 'lon', 'price_per_sqft'],
        X2_features=['lat', 'lon'],
        estimators=[('k_means_3', KMeans(n_clusters=3))],
        titles=['3 clusters'],
        x_label='Latitude',
        y_label='Longitude',
        z_label='Price/Sqft'):

        self.df = pd.read_pickle(df_path)
        self.X3_features = X3_features
        self.X2_features = X2_features
        self.estimators = estimators
        self.titles = titles
        self.x_label='Latitude'
        self.y_label='Longitude'
        self.z_label='Price/Sqft'
        self.model=None


    def three_dimension_plot(self, rect=[0, 0, 1, 1], elev=48, azim=134):

        X = self.df[self.X3_features].values
        fignum = 1

        for name, est in self.estimators:
            fig = plt.figure(fignum, figsize=(12, 12))
            ax = Axes3D(fig, rect=rect, elev=elev, azim=azim)
            est.fit(X)
            self.model = est
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

        return None
    

    def two_dimension_plot(self, rect=[0, 0, 1, 1]):

        X = self.df[self.X2_features].values
        fignum = 1

        for name, est in self.estimators:
            fig, ax = plt.subplots(fignum, figsize=(12, 12))
            est.fit(X)
            self.model = est
            labels = est.labels_

            ax.scatter(X[:, 0], X[:, 1],
                    c=labels.astype(np.float), edgecolor='k')

            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xlabel(self.x_label)
            ax.set_ylabel(self.y_label)
            ax.set_title(self.titles[fignum - 1])
            
            fignum += 1

        fig.show()

        return None
    

    def dump_model(self, filename='joblib_outputs/cluster_vis_model.joblib'):

        dump(self.model, filename)

        return None











