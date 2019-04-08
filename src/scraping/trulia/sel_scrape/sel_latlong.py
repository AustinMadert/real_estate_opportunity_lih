from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import pandas as pd 
import os
import pickle

url = 'https://www.gps-coordinates.net/'

driver = webdriver.Chrome()
driver.get(url)
driver.implicitly_wait(30)

def data_load(pkl_path):
    '''Reads a pickle object and loads pandas dataframe object from a given path
    and then converts a slice of the dataframe for addresses into a list object

    returns: list of addresses
    '''

    df = pd.read_pickle(path)
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

    os.system("echo %s| clip" % text.strip())
    element = driver.find_element_by_xpath(xpath)
    element.send_keys(Keys.CONTROL, 'v')
    return None


def collect_latlongs(addresslist, paste_path, button_path):

    for address in addresslist:
        driver.implicitly_wait(5)
        paste_keys(paste_path, address)
        python_button = driver.find_element_by_xpath(button_path)
        python_button.click()
        

    
    return latlonglist


def main(pkl_path='../trulscraped_data.pkl', 
        paste_path='//input[@id="address"]',
        button_path='//button[@class=btn.btn-primary]'):


    addresslist = data_load(pkl_path)
    latlonglist = collect_latlongs(addresslist, paste_path, button_path)
    data_pickle(latlonglist)



if __name__ == '__main__':
    main()