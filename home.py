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
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold 
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

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
    miner = Miner(db.find_trainingset())
    
    print('done!\nhome is ready! \n\tdict: {"config", "db", "feed_parser", "newsfeed", "miner"}\n')
    
    if meta_classify:
        print('\nmeta-classifing... ')
        miner.meta_classify()

    return {'config': config, 'db': db, 'feed_parser': feed_parser, 'newsfeed': newsfeed, 'miner': miner}


# some util function 
def load_config():
    with open("config/config.json") as f:
        config = json.load(f)
    return config


def likability(article):
    # likability = max(0.5 + (like)*0.5 - (dislike)*0.5 - (not read)*0.2, 0)
    
    if article['dislike']:
        return "DISLIKE"
    elif article['read'] or article['like']:
        return "NOT_DISLIKE"
    return 'IGNORED' # max(likability, 0) (actually this should never happen during training...)


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
                if 'summary' not in e: # if the entry has no summary key just skip it... (it won't happen that often)
                    continue
                soup = BeautifulSoup(e['summary'], features="html.parser")
                imgurl = soup.find('img')

                article_date = datetime.datetime(*e['published_parsed'][:6]) if ('published_parsed' in e) else None
                if article_date is not None and ((datetime.datetime.now() - article_date) > datetime.timedelta(hours=24)):
                    continue  # skip old articles...

                # building the article
                article = {
                    'title' : e['title'] if ('title' in e) else "",
                    'author': e['author'] if ('author' in e) else "",
                    'description' : soup.text if soup is not None else "",
                    'datetime' : article_date.isoformat() if article_date is not None else None,
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
    max_features = 5555
    ignore = lambda x: x # dumb function to ignore function handler that are not needed

    stemmers = {
        'porter': nltk.stem.PorterStemmer,
        'snowball': nltk.stem.SnowballStemmer
    }

    vectorizers = {
        'tfidf': TfidfVectorizer
    }

    classifiers = {
        'multinomial_nb': MultinomialNB,
        'sgd': SGDClassifier,
        'linear_svc': LinearSVC,
        'random_forest': RandomForestClassifier,
        'logistic_regression': LogisticRegression
    }


    def __init__(self, dataset=None):
        self.dataset = pd.DataFrame(dataset)


########## TOKENIZATION #############
    @classmethod
    def remove_stopwords(cls, tokens):
        return [w for w in tokens if w not in cls.stopwords]


    @classmethod
    def word_tokenize(cls, text, ignore_stopwords=False, clean=True):
        
        if clean:
            # replace symbols and multiple spaces with a single space          
            text = re.sub('( +)|(' + r'\W+' + ')', ' ', text)

        tokens = nltk.word_tokenize(text)
        tokens = cls.remove_stopwords(tokens) if ignore_stopwords else tokens

        return tokens


    @classmethod
    def build_article_tokens(cls, article, union=False, ignore_stopwords=False, clean=True):
        t = dict()
        t['title'] = cls.word_tokenize(article['title'], ignore_stopwords, clean)
        t['description'] = cls.word_tokenize(article['description'], ignore_stopwords, clean)
        return list_union(t['title'], t['description']) if union else (t['title']+t['description'])


    @classmethod
    def clean_article_tokens(cls, token):
        regex = re.compile(r'\w+') # remove punctuation
        return [x for x in token if regex.match(x)]


    @classmethod
    def stem_article_tokens(cls, token, stemmer):
        return [stemmer.stem(w) for w in token]


    @classmethod
    def tokenize_article(cls, article, stemmer='snowball',
                         should_union=False, should_ignore_sw=True, should_clean=True, should_stem=True):
        # tokenize article
        tokens = cls.build_article_tokens(article, 
                                            union=should_union, 
                                            ignore_stopwords=should_ignore_sw, 
                                            clean=should_clean)

        return cls.stem_article_tokens(tokens, cls.stemmers[stemmer]('italian')) if should_stem else tokens
########## TOKENIZATION #############


########## CROSS-VALIDATION + META-CLASSIFICATION #############
    @classmethod
    def cross_validate(cls, dataset, labels, classifier, vectorizer, n_class=2, folds=10):
        kf = StratifiedKFold(n_splits=folds)
        total = 0
        totalMat = np.zeros((n_class,n_class))
        
        for train_index, test_index in kf.split(dataset,labels):
            X_train = [dataset[i] for i in train_index]
            X_test = [dataset[i] for i in test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            train_features = vectorizer.fit_transform(X_train) 
            test_features = vectorizer.transform(X_test)

            classifier.fit(train_features,y_train)
            result = classifier.predict(test_features)
            
            mat = confusion_matrix(y_test, result)
            # print(classification_report(y_test, result, target_names=get_categories()))

            totalMat = totalMat + mat
            total = total + sum(y_test==result)
            
        return (totalMat, total/len(labels))


    def meta_classify(self):
        print('\nTokenizing the articles in the dataset...')
        ds = [self.tokenize_article(a) for _,a in self.dataset.iterrows()]
        labels = self.dataset['tag'].to_numpy()

        # preparing modules of the classifier
        print('Preparing the modules (vectorizer and list of classifiers)\n')
        vect = Miner.vectorizers['tfidf'](max_features=Miner.max_features, tokenizer=Miner.ignore, preprocessor=Miner.ignore, token_pattern=None)
        classifiers = [
            ('Multinomial Naive-Bayes', Miner.classifiers['multinomial_nb']()),
            ('Linear SVC (Support Vector Machine)', Miner.classifiers['linear_svc']()),
            ('Random Forest', Miner.classifiers['random_forest'](n_estimators=200, max_depth=3, random_state=42)),
            ('Logistic Regression', Miner.classifiers['logistic_regression'](solver='lbfgs', multi_class='auto', random_state=42)),
            ('SGD Classifeir', Miner.classifiers['sgd'](loss='hinge', penalty='l2',     
                                                        alpha=1e-3, random_state=42,
                                                        max_iter=5, tol=None))
        ]

        # cross validating the classifier
        for c in classifiers:
            print('\n---------------------------')
            print('\nEvaluating ' + c[0] + '\n')
            score = Miner.cross_validate(ds, labels, classifier=c[1], vectorizer=vect, n_class=len(get_categories()))
            print(score)

        print('\n---------------------------')
        print('\nMeta-Classification Done!\n')
        return

#####################   MODEL BUILDING   ##########################
    def learn_vocabulary(self, vectorizer, extract_features=False):
        articles_tokens = [Miner.tokenize_article(a) for _,a in self.dataset.iterrows()]
        
        if extract_features:
            return vectorizer.fit_transform(articles_tokens)
        return vectorizer.fit(articles_tokens)

    def features_from_dataset(self, vectorizer, as_array=True):
        features = self.learn_vocabulary(vectorizer, extract_features=True)
        return features.toarray() if as_array else features


    @classmethod
    def features_from_articles_list(cls, articles, vectorizer, as_array=True):
        articles_tokens = [cls.tokenize_article(a) for a in articles]
        features = vectorizer.transform(articles_tokens)
        return features.toarray() if as_array else features

    # this is the actual classifier (for app)
    def build_news_classifier(self, classifier):
        # this function should provide a pipeline trained object 
        # that i use to fit with the features extracted from the miner
        pass

    def build_likability_predictor(self):
        labels = [likability(a) for _,a in self.dataset.iterrows()]
        return labels
#####################   MODEL BUILDING   ##########################


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
        stats = {
            'num_likes': self.db.articles.count({'like':True, 'read':False}),
            'num_dislikes': self.db.articles.count({'dislike':True, 'read': False}),
            'num_read': self.db.articles.count({'read':True, 'like':False, 'dislike':False}),

            'num_non_dislikes': self.db.articles.count({'like':True, 'read':False}) + self.db.articles.count({'read':True, 'like': False, 'dislike':False}),
            'num_ignored': self.db.articles.count({'like':False, 'dislike':False, 'read':False}),
            'num_read_likes':self.db.articles.count({'read':True, 'like':True}),
            'num_read_dislikes':self.db.articles.count({'read':True, 'dislike':True})
        }
        return stats

    
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