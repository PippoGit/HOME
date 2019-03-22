import json

import feedparser
from bs4 import BeautifulSoup
import hashlib
import datetime

import random

# some util function 
def load_config():
    with open("config/config.json") as f:
        config = json.load(f)
    return config


# NewsFeed class
class NewsFeed:
    def __init__(self, sources):
        self.sources = sources
        self.feed = []


    def load(self):
        for source in self.sources:
            entries = feedparser.parse(source['url']).entries
            for e in entries:
                # parsing the content of the summary
                soup = BeautifulSoup(e['summary'], features="html.parser")
                imgurl = soup.find('img')

                # building the article
                article = {
                    'title' : e['title'] if ('title' in e) else "",
                    'author': e['author'] if ('author' in e) else "",
                    'description' : soup.text if soup is not None else "",
                    'datetime' : e['published'][:-6] if ('published' in e) else "",
                    'img' : imgurl['src'] if imgurl is not None else "",
                    'link': e['link'] if ('link' in e) else "",
                    'source' : source['name']
                }

                # adding the id
                ida = ''.join('{}{}'.format(key, val) for key, val in article.items())
                article['id'] = hashlib.sha1(ida.encode()).hexdigest()
                
                # feed the feeder
                self.feed.append(article)

            print("{} loaded!".format(source['name']))

    
    def sorted_feed(self, num_articles=None):
        feed = self.feed[:num_articles] if num_articles is not None else self.feed
        feed = sorted(feed, key=lambda kv: datetime.datetime.strptime(kv['datetime'], '%a, %d %b %Y %H:%M:%S'), reverse=True)
        return feed


    def training_samples(self, num_articles=50):
        return random.sample(self.feed, num_articles)

# DataMining stuff HERE!
class Miner:
  pass

# MongoDB connector
class DBConnector:
    def __init__(self, host, user, password, name):
        # connecting with mongodb
        self.user = user
        self.host = host
        self.password = password
        self.name = name


    def query(self, query):
        pass

    
    def close(self):
        pass
    

    def insert_feed(self, feed):
        pass


