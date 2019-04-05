
# Adjusted the path to the version of scrapy using exponential distribution autothrottling
import sys
sys.path.append('/Users/austinmadert/scrapy_fork/src/scrapy')

import scrapy
import json
import re
from trulia.items import TruItem

query = 'TX/Austin/'

class TrulSpider(scrapy.Spider):
    name = 'truliaspider'
    allowed_domains = ["trulia.com"]
    start_urls = ['https://www.trulia.com/'+query]

    def last_pagenumber_in_search(self, response):
        try:							
            last_p = response.xpath('//div[@class="txtC"]/a[contains(text(),"Last")]/@href').extract()
            last_p_split = last_p[0].split('/')
            almost_page = last_p_split[-2]
            almost_there = almost_page.split('_p')
            last_page = int(almost_there[0])
            return 134 #last_page
        except:
            return 1

    def parse(self,response):
        last_page_number = self.last_pagenumber_in_search(response)
        page_urls = [response.url + str(pageNumber) +'_p/' for pageNumber in range(1, last_page_number + 1)]

        for page_url in page_urls:
            yield scrapy.Request(page_url, callback=self.parse_listing_results_page)

    
    def parse_listing_results_page(self, response):
        #headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36'}
        for href in response.xpath('//div[@class="containerFluid"]//a/@href').extract():
            url = response.urljoin(href)
            yield scrapy.Request(url, callback=self.parse_listing_contents #, headers=headers
            )

    


    # def json_extract(x):
    # 	x_list = re.split(';',x[0])
    # 	x_json_list=re.split('trulia.pdp.propertyJSON = ',x_list[2])
    # 	return x_json_list[1]

    def parse_listing_contents(self, response):
        item = TruItem()

        #res = response.xpath('//script[@type="text/javascript" and contains(text(),"trulia.pdp.propertyJSON")]').extract()
        # if res:
        #     res_list = re.split(';',res[0])
        #     res_json_list=re.split('trulia.pdp.propertyJSON = ',res_list[2])
        #     json_content = res_json_list[1]
            
            # trulia_json = json.loads(json_content)
        bbsqft = response.xpath('//ul[@class="man"]/li/text()').getall()
        if bbsqft:
            item['house_type'] = bbsqft[2]
            item['bedrooms'] = bbsqft[0]
            item['bathrooms'] = bbsqft[1]
            item['sqft'] = bbsqft[3]
        else:
            self.logger.warning('No bbsqft item received for %s', response.url)

        price = response.xpath('//span[@data-role="price"]/text()').get().strip()
        if price:
            item['price'] = price
        else:
            self.logger.warning('No price item received for %s', response.url)

        address = response.xpath('//div[@data-role="address"]/text()').get().strip()
        if address:
            item['address'] = address
        else:
            self.logger.warning('No address item received for %s', response.url)

        city_state_zip = response.xpath('//span[@data-role="cityState"]/text()').get().strip()
        if city_state_zip:
            item['city_state_zip'] = city_state_zip
        else:
            self.logger.warning('No city_state_zip item received for %s', response.url)
        
        item['url'] = response.url
        yield item

        