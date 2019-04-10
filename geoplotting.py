#!/usr/bin/env python
# coding: utf-8

from mpl_toolkits.basemap import Basemap
from matplotlib.collections import LineCollection
import matplotlib as mpl
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


def create_geoplot(df, lowerlon=-98.10, upperlon=-97.47, lowerlat = 30.11, upperlat = 30.55,
        shapefile_dir = "data/cb_2017_us_zcta510_500k", sample_data=True, sample_size=0.1):

    if sample_data:
        X_train, X_test = train_test_split(df, test_size=sample_size, stratify=df['labels'])

    # Read in data.
    colormap = plt.cm.Purples 

    os.chdir(shapefile_dir)

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

    shp_info = m.readshapefile(os.path.basename(shapefile_dir), 'state')

    lats = X_test['lat'].to_numpy()
    lons = X_test['lon'].to_numpy()
    labels = X_test['labels'].to_numpy()

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

    m.drawrivers(linewidth=1, color='b')

    fig.show()

    return None

