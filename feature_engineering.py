#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
pd.set_option('display.max_rows', 700)
pd.set_option('display.max_columns', 700)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import re
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


pricedf = pd.read_pickle('src/scraping/trulia/sel_scrape/trulscraped_df.pkl')
geodf = pd.read_csv('data/Austin_addresses.csv')
lldf = pd.read_pickle('src/scraping/trulia/sel_scrape/latlonglist_full.pkl')


# In[3]:


lldf = pd.DataFrame(lldf, columns=['address', 'latitude', 'longitude'])


# In[4]:


pricedf = pd.DataFrame(pricedf, columns=['address', 'bathrooms', 'bedrooms', 'city_state_zip', 'house_type', 'price', 'sqft', 'url', 'price_per_sqft', 'adj_address', 'apts'])


# In[5]:


newdf = lldf.merge(pricedf, left_on='address', right_on='adj_address', how='left')


# In[6]:


newdf.drop(newdf[newdf['bedrooms'] == 'Lot/Land'].index, inplace=True)
newdf.drop(newdf[newdf['bathrooms'] == 'Condo'].index, inplace=True)
newdf.drop(newdf[newdf['sqft'] == 0].index , inplace=True)


# In[7]:


newdf['bathrooms'] = newdf['bathrooms'].map(lambda x: str(x).replace('Bathrooms', '').replace('Bathroom', '').strip()).astype(float)
newdf['bedrooms'] = newdf['bedrooms'].map(lambda x: str(x).replace('Bedrooms', '').replace('Bedroom', '').strip()).astype(float)


# In[8]:


X = newdf[['latitude', 'longitude']]
y = newdf['price_per_sqft']


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, y)


# In[10]:


model = LinearRegression(normalize=True)


# In[11]:


model.fit(X_train, y_train)


# In[12]:


model.score(X_test, y_test)


# In[13]:


fig, ax = plt.subplots(figsize=(12,12))
ax.scatter(X['latitude'], X['longitude'])
ax.set_xlabel('latitude')
ax.set_ylabel('longitude')


# In[200]:


fig, ax = plt.subplots(figsize=(12,12))
ax.scatter(X['latitude'], y)
ax.set_xlabel('latitude')
ax.set_ylabel('per_sq_ft')


# In[201]:


fig, ax = plt.subplots(figsize=(12,12))
ax.scatter(X['longitude'], y)
ax.set_xlabel('longitude')
ax.set_ylabel('per_sq_ft')


# In[9]:


newdf.longitude = newdf.longitude.astype(float)
newdf.latitude = newdf.latitude.astype(float)


# In[10]:


newdf.info()


# In[16]:


# lat and long with color as price
tree = RandomForestRegressor(300)
tree.fit(X_train, y_train)


# In[17]:


tree.score(X_test, y_test)


# In[18]:


X = newdf[['latitude', 'longitude']]
y = newdf['price_per_sqft']
X_train, X_test, y_train, y_test = train_test_split(X, y)
tree = RandomForestRegressor(300)
tree.fit(X_train, y_train)
tree.score(X_test, y_test)


# In[19]:


X1 = newdf[['latitude', 'longitude']]
y1 = newdf['price_per_sqft']
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1)
boost = GradientBoostingRegressor(learning_rate=0.01, n_estimators=800)
boost.fit(X1_train, y1_train)
boost.score(X1_test, y1_test)


# In[20]:


geodf.head()


# In[11]:


geodf.c_zip_code = geodf.c_zip_code.astype(str)
geodf.c_zip_code = geodf.c_zip_code.map(lambda x: x.replace('.0', ''))


# In[12]:


geodf['spaces'] = ' '
geodf['adj_address'] = geodf.c_street_number + geodf.spaces + geodf.c_street + geodf.c_city + geodf.spaces + geodf.c_state + geodf.spaces + geodf.c_zip_code


# In[13]:


cleaningdf = geodf.drop(['address', 'address1', 'address2', 'g_street_number', 'g_street', 'g_city', 'g_state','spaces', 'g_unit', 'g_zip_code'], axis=1).copy()


# In[14]:


bigdf = cleaningdf[pd.notnull(cleaningdf.lat)]


# In[15]:


bigdf.drop_duplicates(subset='adj_address', inplace=True)


# In[23]:


newdf.head()


# In[16]:


bigdf.c_unit = bigdf.c_unit.fillna(-1, inplace=True).copy()
newdf.apts = newdf.apts.fillna(-1, inplace=True).copy()


# In[28]:


bigdf.shape


# In[29]:


bigdf[bigdf['adj_address'].isnull()]


# In[30]:


bigdf.reset_index(inplace=True)


# In[31]:


bigdf.iloc[207,4] = 'AUSTIN'
bigdf.iloc[207,10] = bigdf.iloc[207,1] + ' ' + bigdf.iloc[207, 2] + ' ' + bigdf.iloc[207,4] + ' ' + bigdf.iloc[207,5] + ' ' + bigdf.iloc[207,6]
bigdf.iloc[207,:]


# In[32]:


bigdf.drop(columns='index', inplace=True)


# In[33]:


austindf = bigdf[bigdf['c_city'] == 'AUSTIN'].copy()


# In[34]:


print(austindf.c_city.unique(), austindf.c_state.unique(), austindf.g_county.unique())


# In[35]:


X2 = newdf[['latitude', 'longitude']]
y2 = newdf['price_per_sqft']
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2)
forest = RandomForestRegressor(400)
forest.fit(X2_train, y2_train)
forest.score(X2_test, y2_test)


# In[36]:


austindf['price_per_sqft'] = forest.predict(austindf[['lat', 'lon']])


# In[37]:


austindf.head()


# In[38]:


gboost = GradientBoostingRegressor(learning_rate=0.01, n_estimators=1000)
gboost.fit(X2_train, y2_train)
gboost.score(X2_test, y2_test)


# In[39]:


austindf['price_per_sqft_gboost'] = gboost.predict(austindf[['lat', 'lon']])


# In[40]:


austindf.head()


# In[41]:


newdf.head()


# In[42]:


newdf['zip_code'] = newdf['city_state_zip'].map(lambda x: str(x)[-5:])


# In[64]:


testdf = newdf[['latitude', 'longitude', 'price_per_sqft','price']].copy()
# testdf = pd.concat([testdf, pd.get_dummies(testdf['zip_code'])], axis=1)
# testdf.drop('zip_code', axis=1, inplace=True)
testdf.head()


# In[82]:


# y3 = testdf.pop('price_per_sqft')
# X3 = testdf


#random forest
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=.2)
forest = RandomForestRegressor(400)
forest.fit(X3_train, y3_train)
rf = forest.score(X3_test, y3_test)

#gradient boost
gboost = GradientBoostingRegressor(learning_rate=0.01, n_estimators=1000)
gboost.fit(X3_train, y3_train)
gb = gboost.score(X3_test, y3_test)

print("RF score: {}\nGB score: {}".format(rf, gb))


# In[84]:


def train_compute(train_df, train_X, train_y, compute_df, compute_X, compute_y):
    tx = train_df[train_X]
    ty = train_df[train_y]
    gradboost = GradientBoostingRegressor(learning_rate=0.01, n_estimators=1000)
    gradboost.fit(tx, ty)
    compute_df[compute_y] = gradboost.predict(compute_df[compute_X])
    return None
    


# In[86]:


#compute price
train_compute(newdf, ['latitude', 'longitude', 'price_per_sqft'], 'price', austindf, ['lat', 'lon', 'price_per_sqft'], 'price')


# In[92]:


#compute sqft
austindf['sqft'] = austindf['price'] / austindf['price_per_sqft']


# In[94]:


#compute bathrooms
train_compute(newdf, ['latitude', 'longitude', 'price_per_sqft', 'price', 'sqft'], 'bathrooms', austindf, ['lat', 'lon', 'price_per_sqft', 'price', 'sqft'], 'bathrooms')


# In[95]:


#compute bedrooms
train_compute(newdf, ['latitude', 'longitude', 'price_per_sqft', 'price', 'sqft', 'bathrooms'], 'bedrooms', austindf, ['lat', 'lon', 'price_per_sqft', 'price', 'sqft', 'bathrooms'], 'bedrooms')


# In[97]:


#force bedrooms into int in order to be conservative in estimate
austindf.bedrooms = austindf.bedrooms.astype(int)


# In[99]:


austindf.shape


# In[100]:


austindf.to_pickle('boosted_dataset.pkl')


# In[ ]:




