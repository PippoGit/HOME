# parsing & db stuff
import feedparser
from bs4 import BeautifulSoup
import pymongo

# util
import pandas as pd
import numpy as np
import hashlib, datetime, ssl, random, json, re
from collections import defaultdict

# natural language text processing
import nltk
from nltk.corpus import stopwords

# machine learning
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim


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
    miner = Miner(db.find_trainingset())

    return {'config': config, 'db': db, 'feed_parser': feed_parser, 'newsfeed': newsfeed, 'miner': miner}


# some util function 
def load_config():
    with open("config/config.json") as f:
        config = json.load(f)
    return config


def likability(read=False, like=False, dislike=False):
    likability = 0.5 + (like)*0.5 - (dislike)*0.5 - (not read)*0.2
    return max(likability, 0)


def list_union(lst1, lst2): 
    final_list = list(set(lst1) | set(lst2)) 
    return final_list 


def get_categories():
    return ["Politica",
            "Economia",
            "Scienza",
            "Tecnologia",
            "Cultura",
            "Cronaca",
            "Gossip",
            "Sport",
            "Entertainment"]


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

# CustomVectorizer
class TfidfEmbeddingVectorizer:
    def __init__(self):
        self.word2vec = None
        self.word2weight = None
        self.dim = None # len(word2vec.itervalues().next())


    def setw2v(self, w2v):
        self.word2vec = w2v
        self.dim = len(w2v)


    def fit(self, X, y=None):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])


    def fit_transform(self, X, y=None):
        self.fit(X,y)
        return self.transform(X)


class Miner:
    stopwords = set(stopwords.words('italian'))
    max_features = 5000
    ignore = lambda x: x # dumb function to ignore function handler that are not needed

    stemmer = {
        'porter': nltk.stem.PorterStemmer(),
        'snowball': nltk.stem.SnowballStemmer('italian'),
    }

    vectorizer = {
        'tfidf': TfidfVectorizer(analyzer='word', tokenizer=ignore, preprocessor=ignore, token_pattern=None),
        'w2v-tfidf' : TfidfEmbeddingVectorizer()
    }


    def __init__(self, dataset=None, stemmer='snowball', vectorizer='tfidf'):
        self.dataset = pd.DataFrame(dataset)

        self.stemmer = Miner.stemmer[stemmer]
        self.current_stemmer = stemmer

        self.vectorizer = Miner.vectorizer[vectorizer]
        self.current_vectorizer = vectorizer


    def set_stemmer(self, st):
        self.stemmer = Miner.stemmer[st]
        self.current_stemmer = st


    def set_vectorizer(self, vt):
        self.vectorizer = Miner.vectorizer[vt]
        self.current_vectorizer = vt


    def build_w2v(self, tokens_list):
        model = gensim.models.Word2Vec(tokens_list, size=Miner.max_features)
        w2v = dict(zip(model.wv.index2word, model.wv.syn0))
        self.vectorizer.setw2v(w2v)


    def set_dataset(self, dataset):
        self.dataset = pd.DataFrame(dataset)


    def fix_null(self):
        self.dataset.replace('', np.nan, regex=True, inplace=True)


    @classmethod
    def remove_stopwords(cls, tokens):
        return [w for w in tokens if w not in cls.stopwords]


    @classmethod
    def word_tokenize(cls, text, ignore_stopwords=False, clean=True):
        
        if clean:
            # replace strange characters and multiple spaces with a single space          
            text = re.sub('( +)|(' + r'\W+' + ')', ' ', text)

        tokens = nltk.word_tokenize(text)
        tokens = cls.remove_stopwords(tokens) if ignore_stopwords else tokens

        return tokens


    @classmethod
    def build_article_tokens(cls, article, merge=False, ignore_stopwords=False, clean=True):
        t = dict()
        t['title'] = cls.word_tokenize(article['title'], ignore_stopwords, clean)
        t['description'] = cls.word_tokenize(article['description'], ignore_stopwords, clean)
        return list_union(t['title'], t['description']) if merge else t


    @classmethod
    def clean_article_tokens(cls, token):
        regex = re.compile(r'\w+') # remove punctuation

        return [x for x in token if regex.match(x)]


    def stem_article_tokens(self, token):
        return [self.stemmer.stem(w) for w in token]


    def tokenize_article(self, article, 
                         should_merge=True, should_ignore_sw=True, should_clean=True, should_stem=True):
        # tokenize article
        tokens = Miner.build_article_tokens(article, 
                                            merge=should_merge, 
                                            ignore_stopwords=should_ignore_sw, 
                                            clean=should_clean)

        # stemmatize
        if should_stem:
            tokens = self.stem_article_tokens(tokens)

        return tokens

    # fit and build vocabolary (based on self.dataset)
    def learn_vocabulary(self, extract_features=False):
        articles_tokens = [self.tokenize_article(a) for _,a in self.dataset.iterrows()]

        if self.current_vectorizer is 'w2v-tfidf':
            self.build_w2v(articles_tokens)
        
        if extract_features:
            return self.vectorizer.fit_transform(articles_tokens)
        return self.vectorizer.fit(articles_tokens)


    def features_from_dataset(self, as_array=True):
        features = self.learn_vocabulary(extract_features=True)
        return np.asarray(features) if as_array else features


    def features_from_articles_list(self, articles, as_array=True):
        articles_tokens = [self.tokenize_article(a) for a in articles]
        features = self.vectorizer.transform(articles_tokens)
        return np.asarray(features) if as_array else features


    # this is where you validate the classifiers!
    def meta_classify(self):
        # k fold
        # learn
        # test
        # statistics
        # maybe pipeline stuff?
        pass


    # this is the actual classifier (for app)
    def build_news_classifier(self, classifier):
        pass
        # learn vocabulary

        # test classifier

        # return the model
        

    def build_likability_predictor(self):
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
                'tag': {'$exists':True} # ,
                # '$or': [{'read': True}, {'dislike': True}, {'like': True}],
            }, {
                'title':1, 
                'description':1, 
                'tag':1,
                'like':1, 
                'dislike':1, 
                'read':1
            }
        ))
        return results


    def close(self):
        self.client.close()