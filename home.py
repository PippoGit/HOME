# parsing & db stuff
import feedparser
from bs4 import BeautifulSoup
import pymongo

# util
import pandas as pd
import numpy as np
import hashlib, datetime, ssl, random, json, re

# natural language text processing
import nltk
from nltk.corpus import stopwords

# machine learning
from sklearn.feature_extraction.text import CountVectorizer


def test():
    # importing configuration 
    print("\nimporting config file...") 
    config = load_config()

    # preparing the components
    db = DBConnector(**config['db'])
    feed_parser = Parser(config['feeds']) 
    newsfeed = NewsFeed() 

    # loading initial feed
    print("\nloading feeds...") 
    feed_parser.parse()

    # filtering the dataset using some machinelearning magic...
    miner = Miner(feed_parser.parsed_feed)

    return {'config': config, 'db': db, 'feed_parser': feed_parser, 'newsfeed': newsfeed, 'miner': miner}


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


    def training_samples(self, num_articles=50):
        return random.sample(self.parsed_feed, num_articles)


# NewsFeed class:
class NewsFeed:
    pass

# DataMining stuff HERE!
class Miner:
    stopwords = set(stopwords.words("italian"))

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


    @classmethod
    def remove_stopwords(cls, tokens):
        return [w for w in tokens if w not in cls.stopwords]


    @classmethod
    def word_tokenize(cls, text, stopwords=False):
        tokens = nltk.word_tokenize(text)
        tokens = cls.remove_stopwords(tokens) if stopwords else tokens
        return tokens


    @classmethod
    def build_token(cls, article, merge=False, stopwords=False):
        t = dict()
        t['title'] = cls.word_tokenize(article['title'], stopwords)
        t['description'] = cls.word_tokenize(article['description'], stopwords)
        return (t['title'] + t['description']) if merge else t


    @classmethod
    def clean_token(cls, token):
        regex = re.compile(r'\w+')
        return [x for x in token if regex.match(x)]


    def tokenize(self, merge=False, stopwords=False):
        return [Miner.build_token(a, merge, stopwords) for _,a in self.dataset.iterrows()]


    def extract_features(self):
        # features = []
        tokens = self.tokenize(merge=True, stopwords=True)
        tokens = [Miner.clean_token(t) for t in tokens]
        return tokens


    def preprocess(self):
        tokens = self.tokenize()
        return tokens

    
    def tag_classification(self):
        tokens = self.tokenize()
        return tokens


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
        results = list(articles.find(query))
        
        for o in results: 
            o['datetime'] = str(o['datetime'])

        return results

    def find_one(self, query):
        articles = self.db['articles']
        o = articles.find_one(query)
        o['datetime'] = str(o['datetime'])
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
            return random.choice(articles)


    def close(self):
        pass
    
    def insert_feed(self, feed):
        pass


