# -*- coding: utf-8 -*-

# Define here the models for your spider middleware
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/spider-middleware.html

from scrapy import signals
import random


class TruliaSpiderMiddleware(object):
    # Not all methods need to be defined. If a method is not defined,
    # scrapy acts as if the spider middleware does not modify the
    # passed objects.

    @classmethod
    def from_crawler(cls, crawler):
        # This method is used by Scrapy to create your spiders.
        s = cls()
        crawler.signals.connect(s.spider_opened, signal=signals.spider_opened)
        return s

    def process_spider_input(self, response, spider):
        # Called for each response that goes through the spider
        # middleware and into the spider.

        # Should return None or raise an exception.
        return None

    def process_spider_output(self, response, result, spider):
        # Called with the results returned from the Spider, after
        # it has processed the response.

        # Must return an iterable of Request, dict or Item objects.
        for i in result:
            yield i

    def process_spider_exception(self, response, exception, spider):
        # Called when a spider or process_spider_input() method
        # (from other spider middleware) raises an exception.

        # Should return either None or an iterable of Response, dict
        # or Item objects.
        pass

    def process_start_requests(self, start_requests, spider):
        # Called with the start requests of the spider, and works
        # similarly to the process_spider_output() method, except
        # that it doesn’t have a response associated.

        # Must return only requests (not items).
        for r in start_requests:
            yield r

    def spider_opened(self, spider):
        spider.logger.info('Spider opened: %s' % spider.name)


class UserAgentMiddleware(object):
    """This middleware allows spiders to override the user_agent"""

    def __init__(self, user_agent='Scrapy'):
        self.user_agent = user_agent

    @classmethod
    def from_crawler(cls, crawler):
        o = cls(crawler.settings['USER_AGENT'])
        crawler.signals.connect(o.spider_opened, signal=signals.spider_opened)
        return o

    def spider_opened(self, spider):
        self.user_agent = getattr(spider, 'user_agent', self.user_agent)

    def process_request(self, request, spider):
        if self.user_agent:
            request.headers.setdefault(b'User-Agent', self.user_agent)


class RotateUserAgentMiddleware(UserAgentMiddleware):
    def __init__(self, user_agent=''):
        self.user_agent = user_agent

        #the default user_agent_list composes chrome,I E,firefox,Mozilla,opera,netscape
        #for more user agent strings,you can find it in http://www.useragentstring.com/pages/useragentstring.php
        self.user_agent_list = [\
        'Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:47.0) Gecko/20100101 Firefox/47.0',\
        'Mozilla/5.0 (Macintosh; Intel Mac OS X x.y; rv:42.0) Gecko/20100101 Firefox/42.0',\
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36',\
        'Mozilla/5.0 (iPhone; CPU iPhone OS 10_3_1 like Mac OS X) AppleWebKit/603.1.30 (KHTML, like Gecko) Version/10.0 Mobile/14E304 Safari/602.1',\
        'Mozilla/5.0 (X11; Linux x86_64; rv:49.0) Gecko/20100101 Firefox/49.0',\
       ]

    def process_request(self, request, spider):
        ua = random.choice(self.user_agent_list)
        if ua:
            request.headers.setdefault('User-Agent', ua)

            # Adding logging message here.
            spider.log(u'User-Agent: {} {}'.format(request.headers.get('User-Agent'), request))

class ProxiesMiddleware(object):
    def __init__(self, settings):
        self.proxies = [      
            '192.99.203.93:35289',
            '67.205.174.209:1080',
            '198.199.120.102:1080',
            '68.183.126.25:8888',
            '165.227.215.71:1080',
            '67.205.146.29:1080',
            '165.227.215.62:1080',
            '162.243.107.120:1080',
            '159.203.91.6:1080',
            '159.203.166.41:1080',
            '174.138.54.49:1080',
            '162.243.108.129:1080',
            '138.197.108.5:3128',
            '67.205.149.230:1080',
            '192.241.245.207:1080',
            '216.27.126.86:39072',
            '67.205.132.241:1080',
            '162.243.108.161:1080',
            '74.101.171.218:54321',
            '40.76.78.215:80',
            '68.188.59.198:80',
            '148.153.11.58:39593',
            '207.97.174.134:1080',
            '162.243.108.141:1080',
            '72.87.113.190:39593',
            '23.95.0.140:24890',
            '98.221.88.193:64312',
            '24.0.241.151:64312',
            '104.237.227.198:54321',
            '72.89.243.220:64312',
            '50.253.49.189:54321',
            '66.9.8.123:64312',
            '148.77.34.194:54321',
            '66.244.86.186:64312',
            '72.43.17.222:4455'
            ]

    @classmethod
    def from_crawler(cls, crawler):
        return cls(crawler.settings)

    def process_request(self, request, spider):
        pp = random.choice(self.proxies)
        request.meta['proxy'] = pp

'192.99.203.93:35289',
'67.205.174.209:1080',
'198.199.120.102:1080',
'68.183.126.25:8888',
'165.227.215.71:1080',
'67.205.146.29:1080',
'165.227.215.62:1080',
'162.243.107.120:1080',
'159.203.91.6:1080',
'159.203.166.41:1080',
'174.138.54.49:1080',
'162.243.108.129:1080',
'138.197.108.5:3128',
'67.205.149.230:1080',
'192.241.245.207:1080',
'216.27.126.86:39072',
'67.205.132.241:1080',
'162.243.108.161:1080',
'74.101.171.218:54321',
'40.76.78.215:80',
'68.188.59.198:80',
'148.153.11.58:39593',
'207.97.174.134:1080',
'162.243.108.141:1080',
'72.87.113.190:39593',
'23.95.0.140:24890',
'98.221.88.193:64312',
'24.0.241.151:64312',
'104.237.227.198:54321',
'72.89.243.220:64312',
'50.253.49.189:54321',
'66.9.8.123:64312',
'148.77.34.194:54321',
'66.244.86.186:64312',
'72.43.17.222:4455'