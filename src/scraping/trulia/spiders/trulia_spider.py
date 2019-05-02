
# Adjusted the path to the version of scrapy using exponential distribution autothrottling
import sys
sys.path.append('/Users/austinmadert/scrapy_fork/src/scrapy')

import scrapy
from scrapy.crawler import CrawlerProcess
import json
import re
from items import TruItem


query = 'TX/Austin/'


class TrulSpider(scrapy.Spider):
    name = 'truliaspider'
    allowed_domains = ["trulia.com"]
    start_urls = ['https://www.trulia.com/'+query]


    def parse(self, response):
        """Finds the number of pages to scrape, generates the URL for each, and 
        finally yields a scrapy request object for each. Yielding a request object
        will schedule it for scraping.
        
        Arguments:
            response {response object} -- a response from the start URL
        
        Returns:
            None
        """

        last_page_number = 135

        #list of pages to scrape
        page_urls = [response.url + str(pageNumber) +'_p/' for pageNumber in range(1, last_page_number + 1)]

        #for each page, schedule the request and set the callback
        for page_url in page_urls:
            yield scrapy.Request(page_url, callback=self.parse_listing_results_page)

    
    def parse_listing_results_page(self, response):
        """Takes the response for each listing results page and parses for each 
        of the individual listing page urls and then schedules them.
        
        Arguments:
            response {response object} -- a response from the listing results page
        
        Returns:
            None
        """
        
        #take all the listing links and schedule them, setting the callback
        for href in response.xpath('//div[@class="containerFluid"]//a/@href').extract():
            url = response.urljoin(href)
            yield scrapy.Request(url, callback=self.parse_listing_contents
            )

    
    def parse_listing_contents(self, response):
        """Parses the listing page for all the fields in TruItem and uses 
        try/excepts to either store the information in the appropriate TruItem
        field or else if the data was missing store None. Finally, yield the item
        so the pipelines functions can store it in a csv output.
        
        Arguments:
            response {response object} -- a response from the listing page
        
        Returns:
            None
        """
        
        item = TruItem()

        bbsqft = response.xpath('//ul[@class="man"]/li/text()').getall()
        if bbsqft:
            try:
                item['house_type'] = bbsqft[2]
            except:
                item['house_type'] = None
            try:
                item['bedrooms'] = bbsqft[0]
            except:
                item['bedrooms'] = None
            try:
                item['bathrooms'] = bbsqft[1]
            except:
                item['bathrooms'] = None
            try:    
                item['sqft'] = bbsqft[3]
            except:
                item['sqft'] = None

        else:
            self.logger.warning('No bbsqft item received for %s', response.url)

        price = response.xpath('//span[@data-role="price"]/text()').get().strip()
        if price:
            try:
                item['price'] = price
            except:
                item['price'] = None
        else:
            self.logger.warning('No price item received for %s', response.url)

        address = response.xpath('//div[@data-role="address"]/text()').get().strip()
        if address:
            try:
                item['address'] = address
            except:
                item['address'] = None
        else:
            self.logger.warning('No address item received for %s', response.url)

        city_state_zip = response.xpath('//span[@data-role="cityState"]/text()').get().strip()
        if city_state_zip:
            try:
                item['city_state_zip'] = city_state_zip
            except:
                item['city_state_zip'] = None
        else:
            self.logger.warning('No city_state_zip item received for %s', response.url)
        
        item['url'] = response.url
        yield item
