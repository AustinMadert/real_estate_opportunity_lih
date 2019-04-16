#!/usr/bin/env python
# coding: utf-8

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split


def create_geoplot(df, lowerlon=-98.10, upperlon=-97.47, lowerlat = 30.11, upperlat = 30.55,
        shapefile_dir = "/Users/austinmadert/galvanize_repositories/real_estate_opportunity_lih/data/cb_2017_us_zcta510_500k",
        sample_data=True, sample_size=0.1, colors=['r']):

    if sample_data:
        X_train, X_test = train_test_split(df, test_size=sample_size, stratify=df['labels'])
    else:
        X_test = df

    # Read in data.
    colormap = plt.cm.Purples 

    cwd = os.getcwd()
    if cwd != shapefile_dir:
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
    m.fillcontinents(color='#216309', alpha=0.35)

    lats = X_test['lat'].to_numpy()
    lons = X_test['lon'].to_numpy()
    #labels = X_test['labels'].to_numpy()

    x, y = m(lons, lats)
    plt.scatter(x, y, 12, marker='o', color=colors)

    m.drawrivers(linewidth=1, color='b')

    fig.show()

    return None

