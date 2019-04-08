from selenium import webdriver
#from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
#from bs4 import BeautifulSoup
import pandas as pd 
import os
import pickle
import time
import pyperclip

url = 'https://www.latlong.net/convert-address-to-lat-long.html'

driver = webdriver.Chrome()
driver.get(url)
driver.implicitly_wait(10)

def data_load(pkl_path):
    '''Reads a pickle object and loads pandas dataframe object from a given path
    and then converts a slice of the dataframe for addresses into a list object

    returns: list of addresses
    '''

    df = pd.read_pickle(pkl_path)
    return list(df['adj_address'])


def data_pickle(latlonglist):
    '''Takes a list object of addresses, latitudes, and longitudes and writes
    them to a pickle file for later use
    
    returns: None
    '''

    with open('latlonglist.pkl', 'wb') as f:
        pickle.dump(latlonglist, f)
    return None


def paste_keys(xpath, text):
    '''Takes an xpath for the selenium scraper and the text string and pastes
    the text into the web page element

    returns: None
    '''

    # os.system("echo %s| clip" % text.strip())
    element = driver.find_element_by_xpath(xpath)
    for i in range(8):
        element.send_keys(Keys.BACKSPACE)
    element.send_keys(text)
    # element.send_keys(Keys.CONTROL, 'a')
    # element.send_keys(Keys.CONTROL, 'v')
    return None


def collect_latlongs(addresslist, paste_path, button_path, latlong_path, 
                wait=2):

    latlonglist = []

    for address in addresslist:
        time.sleep(wait)

        #paste the address and click the get button"
        paste_keys(paste_path, address)
        get_button = driver.find_element_by_xpath(button_path)
        get_button.click()

        latlong = driver.find_element_by_xpath(latlong_path)
        

        #copy the lat and long into the result list
        lat_field = driver.find_element_by_xpath(lat_path)
        lat_field.send_keys(Keys.CONTROL, 'a')#highlight contents of lat box
        lat_field.send_keys(Keys.CONTROL, 'c')#copy contents of lat box
        latitude = lat_field.send_keys(pyperclip.paste()) #store contents in var

        long_field = driver.find_element_by_xpath(long_path)
        long_field.send_keys(Keys.CONTROL, 'a')#highlight contents of long box
        long_field.send_keys(Keys.CONTROL, 'c')#copy contents of long box
        longitude = long_field.send_keys(pyperclip.paste()) #store contents

        latlonglist.append((latitude, longitude))
    
    return latlonglist


def main(pkl_path='/Users/austinmadert/galvanize_repositories/\
real_estate_opportunity_lih/src/scraping/trulia/sel_scrape/trulscraped_df.pkl', 
        paste_path='//input[@id="JV3220"]',
        button_path='//button[@title="Find_lat_long_coordinates"]',
        latlong_path='//input[@id="latitude" and @class="form-control"]/text()'):


    addresslist = data_load(pkl_path)
    latlonglist = collect_latlongs(addresslist, paste_path, button_path,
                    latlong_path)
    data_pickle(latlonglist)



if __name__ == '__main__':
    main()