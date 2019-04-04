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
from sklearn.feature_extraction.text import TfidfVectorizer


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
    miner = Miner(db.find())

    return {'config': config, 'db': db, 'feed_parser': feed_parser, 'newsfeed': newsfeed, 'miner': miner}


# some util function 
def load_config():
    with open("config/config.json") as f:
        config = json.load(f)
    return config


def likability(read=False, like=False, dislike=False):
    likability = 0.5 + (like)*0.5 - (dislike)*0.5 - (not read)*0.2
    return max(likability, 0)


# useless
def json_serial(obj):
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


# NewsFeed class: (useless)
class NewsFeed:
    pass

# DataMining stuff HERE!
class Miner:
    stopwords = set(stopwords.words("italian"))

    stemmer = {
        'porter': nltk.stem.PorterStemmer()
    }

    vectorizer = {
        'tf-idf': TfidfVectorizer(analyzer='word', tokenizer=lambda doc: doc, preprocessor=lambda doc: doc, token_pattern=None)
    }

    def __init__(self, dataset=None, stemmer='porter', vectorizer='tf-idf'):
        self.dataset = pd.DataFrame(dataset)

        self.model = None

        self.stemmer = Miner.stemmer[stemmer]
        self.vectorizer = Miner.vectorizer[vectorizer]


    def set_stemmer(self, st):
        self.stemmer = Miner.stemmer[st]


    def set_vectorizer(self, vt):
        self.vectorizer = Miner.vectorizer[vt]


    def set_tag_classifier(self):
        pass


    def set_likability_predictor(self):
        pass


    def get_model(self):
        pass


    def set_dataset(self, dataset):
        self.dataset = pd.DataFrame(dataset)


    def fix_null(self):
        self.dataset.replace('', np.nan, regex=True, inplace=True)


    @classmethod
    def tfidf_vectorizer(cls):
        tfidf = TfidfVectorizer(
            analyzer='word',
            tokenizer=lambda doc: doc,
            preprocessor=lambda doc: doc,
            token_pattern=None)
        return tfidf


    @classmethod
    def remove_stopwords(cls, tokens):
        return [w for w in tokens if w not in cls.stopwords]


    @classmethod
    def word_tokenize(cls, text, ignore_stopwords=False):
        tokens = nltk.word_tokenize(text)
        tokens = cls.remove_stopwords(tokens) if ignore_stopwords else tokens
        return tokens


    @classmethod
    def build_token(cls, article, merge=False, ignore_stopwords=False):
        t = dict()
        t['title'] = cls.word_tokenize(article['title'], ignore_stopwords)
        t['description'] = cls.word_tokenize(article['description'], ignore_stopwords)
        return (t['title'] + t['description']) if merge else t


    @classmethod
    def clean_token(cls, token):
        regex = re.compile(r'\w+')
        return [x for x in token if regex.match(x)]


    def stem_token(self, token):
        return [self.stemmer.stem(w) for w in token]


    def features_from_article(self, article):
        # tokenize article
        token = Miner.build_token(article, merge=True, ignore_stopwords=True)
        token = Miner.clean_token(token)

        # stemmatize
        token = self.stem_token(token)

        # vectorize
        ft = self.vectorizer.fit_transform(token)
        return ft.toarray()


    def features_from_dataset(self):
        features = [self.features_from_article(a) for _,a in self.dataset.iterrows()]
        return np.asarray(features)


    def tag_classification(self):
        input_features = self.features_from_dataset()
        target = self.dataset['tag']
        return [input_features, target]


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


    def find(self, query=None):
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
        return {} if len(articles) is 0 else random.choice(articles)

    
    def find_trainingset(self):
        articles = self.db['articles']
        results = list(articles.find({
            'tag': {'$exists':True}}, {
            'title':1, 
            'description':1, 
            'like':1, 
            'dislike':1, 
            'read':1, 
            '_id':0
        }))

        return results


    def close(self):
        self.client.close()