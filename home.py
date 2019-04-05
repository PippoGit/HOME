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
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold 
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

def test(skip_parse=False, meta_classify=False):
    # importing configuration 
    print("\nimporting config file...") 
    config = load_config()

    # preparing the components
    print("\npreparing the components...\n")
    db = DBConnector(**config['db'])
    feed_parser = Parser(config['feeds']) 
    newsfeed = NewsFeed() 

    # loading initial feed
    print("\nloading feeds...") 
    if skip_parse:
        print("actually i'm going to skip this (training purposes)\n")
    else:
        feed_parser.parse()

    # filtering the dataset using some machinelearning magic...
    print("preparing dataset for the miner...\n")
    Miner.max_features = 1000
    miner = Miner(db.find_trainingset(), vectorizer='tfidf')
    
    print('done!\nhome is ready! \n\tdict: {"config", "db", "feed_parser", "newsfeed", "miner"}\n')
    
    if meta_classify:
        print('\nmeta-classifing... ')
        score = miner.meta_classify()

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


def datetime_to_string(article):
    article['datetime'] = str(article['datetime'])
    return article


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


class Miner:
    stopwords = set(stopwords.words('italian'))
    max_features = 5000
    ignore = lambda x: x # dumb function to ignore function handler that are not needed

    stemmer = {
        'porter': nltk.stem.PorterStemmer(),
        'snowball': nltk.stem.SnowballStemmer('italian')
    }

    vectorizer = {
        'tfidf': TfidfVectorizer(max_features=max_features, analyzer='word', tokenizer=ignore, preprocessor=ignore, token_pattern=None)
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

        return self.stem_article_tokens(tokens) if should_stem else tokens


################### NOT REALLY USEFUL FUNCTIONS ###############################################
    def learn_vocabulary(self, extract_features=False):
        articles_tokens = [self.tokenize_article(a) for _,a in self.dataset.iterrows()]
        
        if extract_features:
            return self.vectorizer.fit_transform(articles_tokens)
        return self.vectorizer.fit(articles_tokens)


    def features_from_dataset(self, as_array=True):
        features = self.learn_vocabulary(extract_features=True)
        return features.toarray() if as_array else features


    def features_from_articles_list(self, articles, as_array=True):
        articles_tokens = [self.tokenize_article(a) for a in articles]
        features = self.vectorizer.transform(articles_tokens)
        return features.toarray() if as_array else features
################### NOT REALLY USEFUL FUNCTIONS ###############################################

    
    def cross_validate_tag(self, classifier, folds=10):
        dataset = [self.tokenize_article(a) for _,a in self.dataset.iterrows()]
        labels = self.dataset['tag'].to_numpy()
        len_dataset = len(labels)
            
        kf = StratifiedKFold(n_splits=folds)
        
        total = 0
        totalMat = np.zeros((9,9))
        
        for train_index, test_index in kf.split(dataset,labels):
            X_train = [dataset[i] for i in train_index]
            X_test = [dataset[i] for i in test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            train_features = self.vectorizer.fit_transform(X_train) 
            test_features = self.vectorizer.transform(X_test)
            
            classifier.fit(train_features,y_train)
            result = classifier.predict(test_features)
            
            totalMat = totalMat + confusion_matrix(y_test, result)
            total = total + sum(y_test==result)
            
        return (totalMat, total/len_dataset)


    # this is where you validate the classifiers!
    def meta_classify(self):
        classifier = MultinomialNB()
        score = self.cross_validate_tag(classifier)
        print(score)
        return score


    # this is the actual classifier (for app)
    def build_news_classifier(self, classifier):
        pass

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
        # returns the articles list, but first convert datetime to string to avoid 
        # json parsing issues (or whatever)
        results = [datetime_to_string(o) for o in list(articles.find(query))]
        return results


    def find_one(self, query):
        articles = self.db['articles']
        return datetime_to_string(articles.find_one(query)) # article


    def find_liked(self):
        return self.find({'like':True})[::-1]


    def find_disliked(self):
        return self.find({'dislike':True})[::-1]


    def find_read(self):
        return self.find({'read':True})[::-1]

    
    def find_ignored(self):
        return self.find({'like':False, 'read':False, 'dislike':False})


    def find_untagged(self):
        articles = self.find({'tag':None})
        return {} if len(articles) is 0 else random.choice(articles)

    
    def find_trainingset(self):
        articles = self.db['articles']
        results = list(articles.find({
                'tag': {'$exists':True} # ,
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


    def tag_distribution(self):
        dist = self.db.articles.aggregate([{
                '$group' : {
                    '_id' : {'$ifNull': ['$tag', 'Unknown']},
                    'count': { '$sum': 1 },
                    'num_likes': {'$sum': { '$cond': ["$like", 1, 0] }},
                    'num_dislikes': {'$sum': { '$cond': ["$dislike", 1, 0] }},
                    'num_read': {'$sum': { '$cond': ["$read", 1, 0] }},
                    'num_ignored': {'$sum': { '$cond': [{'$not':{'$or':['$like', '$read', '$dislike']}}, 1, 0] }}
                }
            }])
        return list(dist)


    def like_distribution(self):
        pass

    
    def dislike_distribution(self):
        pass

    
    def read_distribution(self):
        pass


    def stats(self, distribution):
        stats = {
            'tag': self.tag_distribution,
            'like': self.like_distribution,
            'dislike': self.dislike_distribution,
            'read': self.read_distribution
        }
        return stats[distribution]()


    def find_feed(self, descriptor):
        feed_descriptors = {
            'liked': self.find_liked,
            'read': self.find_read,
            'disliked': self.find_disliked,
            'ignored': self.find_ignored,
            'training': self.find_trainingset
        }
        return feed_descriptors[descriptor]()


    def close(self):
        self.client.close()