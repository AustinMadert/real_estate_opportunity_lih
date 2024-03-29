from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
import pandas as pd 
import os
import pickle
import time

url = 'https://www.mapdevelopers.com/geocode_tool.php'

options = Options()
options.add_argument('user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36')
options.add_experimental_option("prefs", {"profile.default_content_settings.cookies": 2})
# options.add_argument('--headless')

driver = webdriver.Chrome(options=options)
driver.set_window_position(-10000,0)
driver.get(url)
driver.implicitly_wait(10)


def data_load(pkl_path):
    '''Reads a pickle object and loads pandas dataframe object from a given path
    and then converts a slice of the dataframe for addresses into a list object

    returns: list of addresses
    '''

    df = pd.read_pickle(pkl_path)
    return list(df['adj_address'])


def data_export(latlonglist):
    '''Takes a list object of addresses, latitudes, and longitudes and writes
    them to a pickle file for later use
    
    returns: None
    '''

    with open('latlonglist_full.pkl', 'wb') as wf:
        pickle.dump(latlonglist, wf)

    return None


def paste_keys(xpath, text):
    '''Takes an xpath for the selenium scraper and the text string and pastes
    the text into the web page element

    returns: None
    '''
    
    element = driver.find_element_by_xpath(xpath)
    for i in range(50):
        element.send_keys(Keys.BACKSPACE)
    element.send_keys(text)
    return None


def collect_latlongs(addresslist, paste_path, button_path, lat_path, long_path,
                wait=5):
    '''Takes a list of addresses and for each address retrieves the latitude
    and longitude from the target website using the xpaths given

    returns: List of tuples with address and its latitude and longitude
    '''

    latlonglist = []
    count = 1

    for address in addresslist:
        
        time.sleep(wait)

        # paste the address and click the get button"
        paste_keys(paste_path, address)
        get_button = driver.find_element_by_xpath(button_path)
        get_button.click()
        time.sleep(2)

        # retrieve latlong text from the webpage
        lat = driver.find_element_by_xpath(lat_path).text
        lon = driver.find_element_by_xpath(long_path).text


        latlonglist.append((address, lat, lon))
        print('Successfully appended try ' + str(count) + ' coordinates')
        count += 1
    
    return latlonglist


def main(pkl_path='/Users/austinmadert/galvanize_repositories/\
real_estate_opportunity_lih/src/scraping/trulia/sel_scrape/trulscraped_df.pkl', 
        paste_path='//div[@class="input-group hidden-xs"]/input[@class="form-control"]',
        button_path='//button[@class="btn btn-default" and @onclick="findAddress();scrollToMap();"]',
        lat_path='//div[@id="display_lat"]',
        long_path='//div[@id="display_lng"]'):

    # load dataframe
    addresslist = data_load(pkl_path)

    # scrape data
    result = collect_latlongs(addresslist, paste_path, button_path,
                    lat_path, long_path)

    data_export(result)


if __name__ == '__main__':
    main()