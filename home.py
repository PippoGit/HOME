import feedparser
from bs4 import BeautifulSoup
import hashlib, datetime, ssl, random, json
import pymongo

import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords


# some util function 
def load_config():
    with open("config/config.json") as f:
        config = json.load(f)
    return config


# useless
def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""

    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    raise TypeError ("Type %s not serializable" % type(obj))
#

# RSS Parser class
class Parser:
    def __init__(self, sources):
        self.sources = sources
        self.parsed_feed = []


    def train(self, miner):
        pass


    def parse(self):
        if hasattr(ssl, '_create_unverified_context'):
            ssl._create_default_https_context = ssl._create_unverified_context

        for source in self.sources:
            parsed = feedparser.parse(source['url'])
            if parsed.bozo == 1:
                print(parsed.bozo_exception)

            entries = parsed.entries
            
            for e in entries:
                # parsing the content of the summary
                soup = BeautifulSoup(e['summary'], features="html.parser")
                imgurl = soup.find('img')

                # building the article
                article = {
                    'title' : e['title'] if ('title' in e) else "",
                    'author': e['author'] if ('author' in e) else "",
                    'description' : soup.text if soup is not None else "",
                    'datetime' : datetime.datetime(*e['published_parsed'][:6]).isoformat() if ('published_parsed' in e) else None,
                    'img' : imgurl['src'] if imgurl is not None else "",
                    'link': e['link'] if ('link' in e) else "",
                    'source' : source['name'],

                    'like' : False,
                    'dislike' : False,
                    'read' : False
                }

                # adding the id
                ida = ''.join('{}{}'.format(key, val) for key, val in article.items())
                article['_id'] = hashlib.sha1(ida.encode()).hexdigest()
                
                # feed the feeder
                self.parsed_feed.append(article)

            print("{} loaded! ({} entries)".format(source['name'], len(entries)))

        print("whole newsfeed loaded ({} entries)".format(len(self.parsed_feed)))


    # IGNORE
    # Still not working, there's too much None values. Just ignore it.
    #   Feed will be sorted by relevance (i hope)
    #   History will be sorted by MongoDB (i hope)
    # def sorted_feed(self, num_articles=None):
    #     return sorted(self.parsed_feed, key=lambda kv: kv['datetime'], reverse=True)
    #
    #
    # def sort_feed(self):
    #     self.parsed_feed = self.sorted_feed()
    # IGNORE 


    def training_samples(self, num_articles=50):
        return random.sample(self.parsed_feed, num_articles)


# NewsFeed class:
class NewsFeed:
    pass

# DataMining stuff HERE!
class Miner:
    def __init__(self, dataset=None):
        self.dataset = [] if dataset is None else pd.DataFrame(dataset)
        self.model = None


    def update_dataset(self, dataset):
        self.training_set = pd.DataFrame(dataset)


    def fix_null(self):
        self.dataset.replace('', np.nan, regex=True, inplace=True)
        print(self.dataset)

    
    def train(self, num_folds=10):
        pass
    

    def get_model(self):
        pass


    def remove_stopwords(self, tokens):
        tokens_nsw = []        
        stop_words=set(stopwords.words("italian"))

        for t in tokens:
            filtered_t = []
            filtered_d = []
            for w in t['title']:
                if w not in stop_words:
                    filtered_t.append(w)
            
            for w in t['description']:
                if w not in stop_words:
                    filtered_d.append(w)
            
            tokens_nsw.append({'title': filtered_t, 'description':filtered_d})
        
        return tokens_nsw

    def tokenize(self, filter=False):
        tokens = []
        for _, article in self.dataset.iterrows():
            t = {'title' : [], 'description': []}
            t['title'] = nltk.word_tokenize(article['title'])
            t['description'] = nltk.word_tokenize(article['description'])
            tokens.append(t) 
        return tokens


    # ????
    def extract_features(self):
        pass


    def preprocess(self):
        tokens = self.tokenize(filter=True)
        return tokens

    
    def classify(self):
        pass


    def build_model(self):
        pass


# MongoDB connector
class DBConnector:
    def __init__(self, host, name, user=None, password=None):
        # connecting with mongodb
        self.client = pymongo.MongoClient(host)
        self.db = self.client[name]

    def update_article(self, article, values):
        articles = self.db['articles']

        results = articles.find_one({'_id':article['_id']})
        if results is None:
            # prepare the new article with updated values
            for k, v in values.items():
                article[k] = v
            
            if article['datetime'] is not None:
                article['datetime'] = datetime.datetime.fromisoformat(article['datetime'])

            # insert article into the db
            articles.insert(article)
        else:
            # update old article
            articles.update(
                {'_id':article['_id']},
                { '$set': values }
            )

    def tag_article(self, article_id, tag):
        self.update_article({'_id':article_id}, {'tag': tag})


    def find(self, query):
        articles = self.db['articles']
        results = articles.find(query)
        return list(results)

    def find_one(self, query):
        articles = self.db['articles']
        return articles.find_one(query)
        
    def find_liked(self):
        return self.find({'like':True})[::-1]

    def find_disliked(self):
        return self.find({'dislike':True})[::-1]

    def find_read(self):
        return self.find({'read':True})[::-1]

    def find_untagged(self):
        articles = self.find({'tag':None})

        if len(articles) is 0:
            return {}
        else:
            article = random.choice(articles)
            article['datetime'] = str(article['datetime'])
            return article

    def close(self):
        pass
    
    def insert_feed(self, feed):
        pass


