#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from math import sin, cos, sqrt, atan2, radians
from pickle import loads
import warnings
warnings.filterwarnings('ignore')
np.random.seed(5)


def data_imports():
    prices = pd.read_pickle('src/scraping/trulia/sel_scrape/trulscraped_df.pkl')
    geodf = pd.read_csv('data/Austin_addresses.csv')
    lls = pd.read_pickle('src/scraping/trulia/sel_scrape/latlonglist_full.pkl')

    lldf = pd.DataFrame(lls, columns=['address', 'latitude', 'longitude'])
    pricedf = pd.DataFrame(prices, columns=['address', 'bathrooms', 'bedrooms', \
        'city_state_zip', 'house_type', 'price', 'sqft', 'url', 'price_per_sqft', \
        'adj_address', 'apts'])
    train_df = lldf.merge(pricedf, left_on='address', right_on='adj_address', how='left')

    return train_df, geodf


def clean_dataframes(train_df, geodf):

    train_df.drop(train_df[train_df['bedrooms'] == 'Lot/Land'].index, inplace=True)
    train_df.drop(train_df[train_df['bathrooms'] == 'Condo'].index, inplace=True)
    train_df.drop(train_df[train_df['sqft'] == 0].index , inplace=True)

    train_df['bathrooms'] = train_df['bathrooms'].map(lambda x: str(x)\
        .replace('Bathrooms', '').replace('Bathroom', '').strip()).astype(float)
    train_df['bedrooms'] = train_df['bedrooms'].map(lambda x: str(x)\
        .replace('Bedrooms', '').replace('Bedroom', '').strip()).astype(float)

    train_df.longitude = train_df.longitude.astype(float)
    train_df.latitude = train_df.latitude.astype(float)

    geodf.drop(['address', 'address1', 'address2',\
         'g_street_number', 'g_street', 'g_city', 'g_state',\
         'g_unit', 'g_zip_code'], axis=1, inplace=True)

    geodf.c_zip_code = geodf.c_zip_code.astype(str)
    geodf.c_zip_code = geodf.c_zip_code.map(lambda x: x.replace('.0', ''))

    bigdf = geodf[pd.notnull(geodf.lat)]

    bigdf.iloc[207,4] = 'AUSTIN'

    bigdf['adj_address'] = geodf.c_street_number + ' ' + geodf.c_street +\
         geodf.c_city + ' ' + geodf.c_state + ' ' + geodf.c_zip_code

    bigdf.drop_duplicates(subset='adj_address', inplace=True)

    bigdf.reset_index(inplace=True)
    bigdf.drop(columns='index', inplace=True)

    city_df = bigdf[bigdf['c_city'] == 'AUSTIN'].copy()

    # get list of coordinates
    print('Getting list of city address coordinates...')
    subset = city_df[['lat', 'lon']]
    tuples = [tuple(x) for x in subset.values]
    distances = [haversine_to_downtown(coord) for coord in tuples]
    city_df['dist_to_downtown'] = distances

    print('Getting list of city bus stop coordinates...')
    stops = gpd.read_file('data/Shapefiles_20-_20JANUARY_202018/Stops/Stops.shp')
    subs = stops[['LATITUDE', 'LONGITUDE']].dropna()
    tups = [tuple(x) for x in subs.values]
    min_transport_dist = [min([two_point_haversine(stop, address) for stop in tups]) for address in tuples]
    city_df['min_dist_to_transport'] = min_transport_dist

    intown_df = city_df[city_df['dist_to_downtown'] < (city_df.dist_to_downtown.std() * 4)].copy()

    pickle_dataframe(train_df, 'train_df.pkl')
    pickle_dataframe(intown_df, 'intown_df.pkl')

    return None


def haversine_to_downtown(point):
    # calculates the distance between two points (lat, lngs) on a great circle, or on the 
    # surface of a sphere (in this case the sphere is planet earth)
    # units in km
    lat, lng = point
    deglen = 110.25
    x = lat - 30.2648
    y = (lng - (-97.7472)) * cos(-97.7472)

    return deglen * sqrt(x*x + y*y)


def two_point_haversine(point1, point2):
    lat1, lng1 = point1
    lat2, lng2 = point2
    deglen = 110.25
    x = lat1 - lat2
    y = (lng1 - (lng2)) * cos(lng2)

    return deglen * sqrt(x*x + y*y)


def train_compute(train_df, train_features, train_labels, compute_df, compute_X, compute_y):

    X = train_df[train_features]
    y = train_df[train_labels]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    gradboost = GradientBoostingRegressor(learning_rate=0.01, n_estimators=1000)
    gradboost.fit(X_train, y_train)
    score = gradboost.score(X_test, y_test)
    compute_df[compute_y] = gradboost.predict(compute_df[compute_X])
    
    return score


def boost_dataframe(gridsearchflag=False):        
    
    train_df = pd.read_pickle('train_df.pkl')
    intown_df = pd.read_pickle('intown_df.pkl')
    
    #compute price/sqft
    price_per_sqft_score = train_compute(
        train_df,
        ['latitude', 'longitude'],
        'price_per_sqft',
        intown_df,
        ['lat', 'lon'],
        'price_per_sqft'
    )

    #compute price
    price_score = train_compute(
        train_df,
        ['latitude', 'longitude', 'price_per_sqft'], 
        'price',
        intown_df, 
        ['lat', 'lon', 'price_per_sqft'], 
        'price'
        )

    #compute sqft
    intown_df['sqft'] = intown_df['price'] / intown_df['price_per_sqft']

    #compute bathrooms
    bathroom_score = train_compute(
        train_df, 
        ['latitude', 'longitude', 'price_per_sqft', 'price', 'sqft'],
        'bathrooms', 
        intown_df, 
        ['lat', 'lon', 'price_per_sqft', 'price', 'sqft'], 
        'bathrooms'
        )

    #compute bedrooms
    bedroom_score = train_compute(
        train_df, 
        ['latitude', 'longitude', 'price_per_sqft', 'price', 'sqft', 'bathrooms'],
        'bedrooms', 
        intown_df, 
        ['lat', 'lon', 'price_per_sqft', 'price', 'sqft', 'bathrooms'], 
        'bedrooms'
        )

    #force bedrooms into int in order to be conservative in estimate
    intown_df.bedrooms = intown_df.bedrooms.astype(int)
    intown_df['price_per_bed'] = intown_df['price'] / intown_df['bedrooms']

    print('Price/Sqft score: {}\nPrice score: {}\nBathroom score: {}\nBedroom score: {}'\
        .format(price_per_sqft_score, price_score, bathroom_score, bedroom_score))

    return intown_df

def searcher(df, parameter_space):

    return clf.get_params

def pickle_dataframe(df, name):

    df.to_pickle(name)

    return None


def main():
    response = input('Would like to skip cleaning and use pickle? ')
    if response == 'y':
        train_df, geo_df = data_imports()
        clean_dataframes(train_df, geo_df)
        boosted_aus = boost_dataframe()
        pickle_dataframe(boosted_aus, 'boosted_dataset.pkl')
    else:
        response2 = input('Use gridsearch? ')
        if response2 == 'y':
            boosted_aus = boost_dataframe(gridsearchflag=True)
            pickle_dataframe(boosted_aus, 'boosted_dataset.pkl')
        else:
            boosted_aus = boost_dataframe()
            pickle_dataframe(boosted_aus, 'boosted_dataset.pkl')
    
    return None



if __name__ == '__main__':
    main()



