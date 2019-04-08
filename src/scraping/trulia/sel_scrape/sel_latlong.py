from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

url = 'https://www.gps-coordinates.net/'

browser = webdriver.Chrome()
browser.get(url)