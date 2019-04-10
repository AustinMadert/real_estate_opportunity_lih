#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import geopandas as gpd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from math import sin, cos, sqrt, atan2, radians


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

    geodf.c_zip_code = geodf.c_zip_code.astype(str)
    geodf.c_zip_code = geodf.c_zip_code.map(lambda x: x.replace('.0', ''))

    geodf['spaces'] = ' '
    geodf['adj_address'] = geodf.c_street_number + geodf.spaces +\
        geodf.c_street + geodf.c_city + geodf.spaces + geodf.c_state +\
        geodf.spaces + geodf.c_zip_code

    cleaningdf = geodf.drop(['address', 'address1', 'address2',\
         'g_street_number', 'g_street', 'g_city', 'g_state','spaces',\
         'g_unit', 'g_zip_code'], axis=1).copy()

    bigdf = cleaningdf[pd.notnull(cleaningdf.lat)]

    bigdf.drop_duplicates(subset='adj_address', inplace=True)

    bigdf[bigdf['adj_address'].isnull()]

    bigdf.reset_index(inplace=True)
    bigdf.drop(columns='index', inplace=True)

    bigdf.iloc[207,4] = 'AUSTIN'
    bigdf.iloc[207,10] = bigdf.iloc[207,1] + ' ' + bigdf.iloc[207, 2] +\
         ' ' + bigdf.iloc[207,4] + ' ' + bigdf.iloc[207,5] +\
         ' ' + bigdf.iloc[207,6]

    austin_df = bigdf[bigdf['c_city'] == 'AUSTIN'].copy()

    # get list of coordinates
    subset = austin_df[['lat', 'lon']]
    tuples = [tuple(x) for x in subset.values]
    distances = [haversine_to_downtown(coord) for coord in tuples]
    austin_df['dist_to_downtown'] = distances

    stops = gpd.read_file('data/Shapefiles_20-_20JANUARY_202018/Stops/Stops.shp')
    subs = stops[['LATITUDE', 'LONGITUDE']]
    tups = [tuple(x) for x in subs.values]
    min_transport_dist = [min([two_point_haversine(stop, address) for stop in tups]) for address in tuples]
    austin_df['min_dist_to_transport'] = min_transport_dist

    return train_df, austin_df


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


def boost_dataframe(train_df, austin_df):        
    #compute price
    price_score = train_compute(
        train_df,
        ['latitude', 'longitude', 'price_per_sqft'], 
        'price',
        austin_df, 
        ['lat', 'lon', 'price_per_sqft'], 
        'price'
        )

    #compute sqft
    austin_df['sqft'] = austin_df['price'] / austin_df['price_per_sqft']

    #compute bathrooms
    bathroom_score = train_compute(
        train_df, 
        ['latitude', 'longitude', 'price_per_sqft', 'price', 'sqft'],
        'bathrooms', 
        austin_df, 
        ['lat', 'lon', 'price_per_sqft', 'price', 'sqft'], 
        'bathrooms'
        )

    #compute bedrooms
    bedroom_score = train_compute(
        train_df, 
        ['latitude', 'longitude', 'price_per_sqft', 'price', 'sqft', 'bathrooms'],
        'bedrooms', 
        austin_df, 
        ['lat', 'lon', 'price_per_sqft', 'price', 'sqft', 'bathrooms'], 
        'bedrooms'
        )

    #force bedrooms into int in order to be conservative in estimate
    austin_df.bedrooms = austin_df.bedrooms.astype(int)
    austin_df['price_per_bed'] = austin_df['price'] / austin_df['bedrooms']

    print('Price score: {}\nBathroom score: {}\nBedroom score: {}'\
        .format(price_score, bathroom_score, bedroom_score))

    return austin_df


def pickle_dataframe(df, name):

    df.to_pickle(name)

    return None


def main():
    train_df, geo_df = data_imports()
    clean_train, clean_aus = clean_dataframes(train_df, geo_df)
    boosted_aus = boost_dataframe(clean_train, clean_aus)
    pickle_dataframe(boosted_aus, 'boosted_dataset.pkl')
    return None



if __name__ == '__main__':
    main()



