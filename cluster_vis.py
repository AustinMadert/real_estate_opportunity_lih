#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from sklearn.cluster import KMeans
from math import sin, cos, sqrt, atan2, radians
from mpl_toolkits.mplot3d import Axes3D
np.random.seed(5)

df = pd.read_pickle('boosted_dataset.pkl')

def haversine_to_downtown(point):
    # calculates the distance between two points (lat, lngs) on a great circle, or on the 
    # surface of a sphere (in this case the sphere is planet earth)
    # units in km
    lat, lng = point
    deglen = 110.25
    x = lat - 30.2648
    y = (lng - (-97.7472))*cos(-97.7472)
    return deglen*sqrt(x*x + y*y)

def two_point_haversine(point1, point2):
    lat1, lng1 = point1
    lat2, lng2 = point2
    deglen = 110.25
    x = lat1 - lat2
    y = (lng1 - (lng2))*cos(lng2)
    return deglen*sqrt(x*x + y*y)

# get list of coordinates
subset = df[['lat', 'lon']]
tuples = [tuple(x) for x in subset.values]

distances = [haversine_to_downtown(coord) for coord in tuples]

df['dist_to_downtown'] = distances

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

df.drop('price_per_sqft_gboost', axis=1, inplace=True)

df['price_per_bed'] = df['price'] / df['bedrooms']

stops = gpd.read_file('data/Shapefiles_20-_20JANUARY_202018/Stops/Stops.shp')

subs = stops[['LATITUDE', 'LONGITUDE']]
tups = [tuple(x) for x in subs.values]

min_transport_dist = [min([two_point_haversine(stop, address) for stop in tups]) for address in tuples]

df['min_dist_to_transport'] = min_transport_dist

df.to_pickle('out_from_cluster_into_geoplotting.pkl')

df = pd.read_pickle('out_from_cluster_into_geoplotting.pkl')

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




estimators = [('k_means_4', KMeans(n_clusters=4)),
              ('k_means_3', KMeans(n_clusters=3)),
              ('k_means_2', KMeans(n_clusters=2))]

fignum = 1
titles = ['4 clusters', '3 clusters', '2 clusters']
for name, est in estimators:
    fig = plt.figure(fignum, figsize=(12, 12))
    ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=32, azim=70)
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






