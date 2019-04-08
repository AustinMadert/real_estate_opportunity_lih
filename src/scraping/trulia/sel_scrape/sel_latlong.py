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

def data_load():
    pass

def data_pickle():
    pass

def paste_keys(self, xpath, text):
    os.system("echo %s| clip" % text.strip())
    el = self.driver.find_element_by_xpath(xpath)
    el.send_keys(Keys.CONTROL, 'v')

def main():
    pass


if __name__ == '__main__':
    main()