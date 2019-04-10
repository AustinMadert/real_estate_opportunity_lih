#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import re
import matplotlib.pyplot as plt


def data_imports():
    prices = pd.read_pickle('src/scraping/trulia/sel_scrape/trulscraped_df.pkl')
    geodf = pd.read_csv('data/Austin_addresses.csv')
    lls = pd.read_pickle('src/scraping/trulia/sel_scrape/latlonglist_full.pkl')

    lldf = pd.DataFrame(lls, columns=['address', 'latitude', 'longitude'])
    pricedf = pd.DataFrame(prices, columns=['address', 'bathrooms', 'bedrooms', \
        'city_state_zip', 'house_type', 'price', 'sqft', 'url', 'price_per_sqft', \
        'adj_address', 'apts'])
    newdf = lldf.merge(pricedf, left_on='address', right_on='adj_address', how='left')

    return [lldf, pricedf, newdf]

def clean_dataframes():
    

newdf.drop(newdf[newdf['bedrooms'] == 'Lot/Land'].index, inplace=True)
newdf.drop(newdf[newdf['bathrooms'] == 'Condo'].index, inplace=True)
newdf.drop(newdf[newdf['sqft'] == 0].index , inplace=True)

newdf['bathrooms'] = newdf['bathrooms'].map(lambda x: str(x).replace('Bathrooms', '').replace('Bathroom', '').strip()).astype(float)
newdf['bedrooms'] = newdf['bedrooms'].map(lambda x: str(x).replace('Bedrooms', '').replace('Bedroom', '').strip()).astype(float)

X = newdf[['latitude', 'longitude']]
y = newdf['price_per_sqft']
X_train, X_test, y_train, y_test = train_test_split(X, y)

newdf.longitude = newdf.longitude.astype(float)
newdf.latitude = newdf.latitude.astype(float)

X1 = newdf[['latitude', 'longitude']]
y1 = newdf['price_per_sqft']
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1)
boost = GradientBoostingRegressor(learning_rate=0.01, n_estimators=800)
boost.fit(X1_train, y1_train)
boost.score(X1_test, y1_test)

geodf.c_zip_code = geodf.c_zip_code.astype(str)
geodf.c_zip_code = geodf.c_zip_code.map(lambda x: x.replace('.0', ''))

geodf['spaces'] = ' '
geodf['adj_address'] = geodf.c_street_number + geodf.spaces + geodf.c_street + geodf.c_city + geodf.spaces + geodf.c_state + geodf.spaces + geodf.c_zip_code

cleaningdf = geodf.drop(['address', 'address1', 'address2', 'g_street_number', 'g_street', 'g_city', 'g_state','spaces', 'g_unit', 'g_zip_code'], axis=1).copy()

bigdf = cleaningdf[pd.notnull(cleaningdf.lat)]

bigdf.drop_duplicates(subset='adj_address', inplace=True)

bigdf.c_unit = bigdf.c_unit.fillna(-1, inplace=True).copy()
newdf.apts = newdf.apts.fillna(-1, inplace=True).copy()

bigdf[bigdf['adj_address'].isnull()]

bigdf.reset_index(inplace=True)

bigdf.iloc[207,4] = 'AUSTIN'
bigdf.iloc[207,10] = bigdf.iloc[207,1] + ' ' + bigdf.iloc[207, 2] + ' ' + bigdf.iloc[207,4] + ' ' + bigdf.iloc[207,5] + ' ' + bigdf.iloc[207,6]
bigdf.iloc[207,:]

bigdf.drop(columns='index', inplace=True)

austindf = bigdf[bigdf['c_city'] == 'AUSTIN'].copy()

testdf = newdf[['latitude', 'longitude', 'price_per_sqft','price']].copy()

def train_compute(train_df, train_X, train_y, compute_df, compute_X, compute_y):
    tx = train_df[train_X]
    ty = train_df[train_y]
    gradboost = GradientBoostingRegressor(learning_rate=0.01, n_estimators=1000)
    gradboost.fit(tx, ty)
    compute_df[compute_y] = gradboost.predict(compute_df[compute_X])
    return None

def boost_dataframe():        
    #compute price
    train_compute(newdf, ['latitude', 'longitude', 'price_per_sqft'], 'price', austindf, ['lat', 'lon', 'price_per_sqft'], 'price')
    #compute sqft
    austindf['sqft'] = austindf['price'] / austindf['price_per_sqft']
    #compute bathrooms
    train_compute(newdf, ['latitude', 'longitude', 'price_per_sqft', 'price', 'sqft'], 'bathrooms', austindf, ['lat', 'lon', 'price_per_sqft', 'price', 'sqft'], 'bathrooms')
    #compute bedrooms
    train_compute(newdf, ['latitude', 'longitude', 'price_per_sqft', 'price', 'sqft', 'bathrooms'], 'bedrooms', austindf, ['lat', 'lon', 'price_per_sqft', 'price', 'sqft', 'bathrooms'], 'bedrooms')
    #force bedrooms into int in order to be conservative in estimate
    austindf.bedrooms = austindf.bedrooms.astype(int)
    return austindf

def pickle_dataframe(df, name):

    df.to_pickle(name)

    return None

def main():
    data_imports()
    



if __name__ == '__main__':
    main()



