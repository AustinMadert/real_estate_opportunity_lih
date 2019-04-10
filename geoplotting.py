#!/usr/bin/env python
# coding: utf-8



from mpl_toolkits.basemap import Basemap
from matplotlib.collections import LineCollection
import matplotlib as mpl
from matplotlib.colors import rgb2hex
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.cluster import KMeans
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.cm as cm
import matplotlib.colors as colors
import geopandas as gpd
from sklearn.model_selection import train_test_split


df = pd.read_pickle('boosted_dataset.pkl')




cluster = KMeans(n_clusters=3)




X = df[['min_dist_to_transport', 'price_per_bed', 'price_per_sqft']]




cluster.fit(X)




df['labels'] = cluster.labels_




X_train, X_test = train_test_split(df, test_size=0.01, stratify=df['labels'])




df.head()




# # Read in data.
# colormap = plt.cm.Purples 

# aus_stop_file_dir = "data/cb_2017_us_zcta510_500k"
# os.chdir(aus_stop_file_dir)

# # Austin coordinates.
# lowerlon = -98.10 
# upperlon = -97.47
# lowerlat = 30.11
# upperlat = 30.55

# fig = plt.figure(figsize=(12,12))
# m = Basemap(
#     llcrnrlon=lowerlon,
#     llcrnrlat=lowerlat,
#     urcrnrlon=upperlon,
#     urcrnrlat=upperlat,
#     projection="lcc",
#     resolution="h",
#     lat_0=lowerlat,
#     lat_1=upperlat,
#     lon_0=lowerlon,
#     lon_1=upperlon
#     )

# shp_info = m.readshapefile(os.path.basename(aus_stop_file_dir), 'state')

# # get list of coordinates
# subset = df[['lat', 'lon', 'labels']]
# tuples = [tuple(x) for x in subset.values]

# for i in tuples:
#     if i[2] == 0:
#         color='b'
#     elif i[2] == 1:
#         color='r'
#     else:
#         color='y'
#     m.plot(i[0], i[1], color=color, markersize=1)
# #     print('plotted')


# m.drawrivers(linewidth=1, color='b')
# ;




# Read in data.
colormap = plt.cm.Purples 

aus_stop_file_dir = "data/cb_2017_us_zcta510_500k"
os.chdir(aus_stop_file_dir)

# Austin coordinates.
lowerlon = -98.10 
upperlon = -97.47
lowerlat = 30.11
upperlat = 30.55

fig = plt.figure(figsize=(12,12))
m = Basemap(
    llcrnrlon=lowerlon,
    llcrnrlat=lowerlat,
    urcrnrlon=upperlon,
    urcrnrlat=upperlat,
    projection="lcc",
    resolution="h",
    lat_0=lowerlat,
    lat_1=upperlat,
    lon_0=lowerlon,
    lon_1=upperlon
    )

shp_info = m.readshapefile(os.path.basename(aus_stop_file_dir), 'state')

lats = X_test['lat'].to_numpy()
lons = X_test['lon'].to_numpy()
labels = X_test['labels'].to_numpy()
# tuples = [tuple(x) for x in subset.values]

colors = []
for i in labels:
    if i == 0:
        colors.append('b')
    elif i == 1:
        colors.append('r')
    else:
        colors.append('y')

x, y = m(lons, lats)
plt.scatter(x, y, 12, marker='o', color=colors)


# for i in tuples:
#     if i[2] == 0:
#         color='b'
#     elif i[2] == 1:
#         color='r'
#     else:
#         color='y'
#     m.scatter(i[0], i[1], color=color, zorder=5)



m.drawrivers(linewidth=1, color='b')
("")




X_train2, X_test2 = train_test_split(df, test_size=0.1, stratify=df['labels'])




# Read in data.
colormap = plt.cm.Purples 

# aus_stop_file_dir = "data/cb_2017_us_zcta510_500k"
# os.chdir(aus_stop_file_dir)

# Austin coordinates.
lowerlon = -98.10 
upperlon = -97.47
lowerlat = 30.11
upperlat = 30.55

fig = plt.figure(figsize=(12,12))
m = Basemap(
    llcrnrlon=lowerlon,
    llcrnrlat=lowerlat,
    urcrnrlon=upperlon,
    urcrnrlat=upperlat,
    projection="lcc",
    resolution="h",
    lat_0=lowerlat,
    lat_1=upperlat,
    lon_0=lowerlon,
    lon_1=upperlon
    )

shp_info = m.readshapefile(os.path.basename(aus_stop_file_dir), 'state')

lats = X_test2['lat'].to_numpy()
lons = X_test2['lon'].to_numpy()
labels = X_test2['labels'].to_numpy()
# tuples = [tuple(x) for x in subset.values]

colors = []
for i in labels:
    if i == 0:
        colors.append('b')
    elif i == 1:
        colors.append('r')
    else:
        colors.append('y')

x, y = m(lons, lats)
plt.scatter(x, y, 12, marker='o', color=colors)


# for i in tuples:
#     if i[2] == 0:
#         color='b'
#     elif i[2] == 1:
#         color='r'
#     else:
#         color='y'
#     m.scatter(i[0], i[1], color=color, zorder=5)



m.drawrivers(linewidth=1, color='b')
("")






