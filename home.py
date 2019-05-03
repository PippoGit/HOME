# parsing & db stuff
import feedparser
from bs4 import BeautifulSoup
import pymongo

# util
import math
import pandas as pd
import numpy as np
import hashlib, datetime, ssl, random, json, re, string
from collections import defaultdict, Counter
from statistics import mean

# plot
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sn
from wordcloud import WordCloud

# natural language text processing
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import ItalianStemmer

from unidecode import unidecode

# machine learning
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, learning_curve
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, f1_score, recall_score, precision_score


from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from mlxtend.classifier import StackingCVClassifier

from sklearn.cluster import KMeans

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.decomposition import TruncatedSVD

from gensim.sklearn_api import D2VTransformer

# SPACY and Stuff...
import spacy
nlp = spacy.load('it_core_news_sm')


def test_classifier():
    print("\nimporting config file...") 
    config = load_config()

    # preparing the components
    print("\npreparing the components...\n")
    db = DBConnector(**config['db'])
    feed_parser = Parser(random.sample(config['feeds'], 5)) # taking just 5 random sources

    # loading initial feed
    print("\nloading feeds...") 
    feed_parser.parse()

    print('\nbuilding the model...')
    miner = Miner(db.find_trainingset())

    model = miner.build_news_classifier()
    X = Miner.tokenize_list(feed_parser.parsed_feed)

    results = model.predict(X)

    for i in range(1, 20):
        print('article ' + str(i) + ': ')
        print(feed_parser.parsed_feed[i])
        print('predicted category: ')
        print(results[i])


def test(skip_parse=False, meta_classify=False, show_mat=True, tuning=False, w2v=False, wordcloud=False):

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
    
    if meta_classify:
        print('\nmeta-classifing... ')
        miner.meta_classify(show_mat=show_mat, tuning=tuning, use_w2v=w2v)

    # WORLDCLOUDSTUFF...
    if wordcloud:
        show_wordcloud(flatten(Miner.tokenize_list(miner.dataset)))

    print('done!\nhome is ready! \n\tdict: {"config", "db", "feed_parser", "newsfeed", "miner"}\n')
    return {'config': config, 'db': db, 'feed_parser': feed_parser, 'newsfeed': newsfeed, 'miner': miner}


# some util function 
def load_config():
    with open("config/config.json") as f:
        config = json.load(f)
    return config


def likability(article):
    likability = max(0.5 + (article['like'])*0.5 - (article['dislike'])*0.5 - (not article['read'])*0.2, 0)
    
    if article['dislike']:
        return ("DISLIKE", likability)
    elif article['read'] or article['like']:
        return ("NOT_DISLIKE", likability)
    return ('IGNORED', likability) # (actually this should never happen during training...)


def list_union(lst1, lst2): 
    final_list = list(set(lst1) | set(lst2)) 
    return final_list 


def get_aggr_categories():
# db.articles.find().forEach(function (article){
#    var new_tag = article.tag;
#    if((['Scienza', 'Tecnologia']).includes(article.tag))
#        new_tag = "Scienza&Tecnologia";
#    else if(['Politica', 'Economia'].includes(article.tag))
#        new_tag = "Politica&Economia";
#    else if(['Gossip', 'Entertainment'].includes(article.tag))
#        new_tag = "Gossip&Entertainment";
    
#    db.articles.update({_id: article._id},{$set:{"tag__": new_tag}});
# })

    return [
        "Cronaca",
        "Cultura",
        "Entertainment",
        "Gossip",
        "Politica&Economia",
        "Scienza&Tecnologia",
        "Sport",
    ]


def get_categories():
    return [
            "Cronaca",
            "Cultura",
            "Economia",
            "Entertainment",
            "Gossip",
            "Politica",
            "Scienza",
            "Sport",
            "Tecnologia"
    ]


def datetime_to_string(article):
    article['datetime'] = str(article['datetime'])
    return article


def try_parse_date(article_date):
    now = datetime.datetime.now()
    dt = datetime.datetime.strptime(article_date, "%Y-%m-%dT%H:%M:%S")  if article_date is not None else datetime.datetime(now.year, now.month, now.day)
    return dt


flatten = lambda l: [item for sublist in l for item in sublist]


def show_wordcloud(dataset):
    word_cloud_dict = Counter(dataset)
    wordcloud = WordCloud(width = 1000, height = 500).generate_from_frequencies(word_cloud_dict)
    plt.figure(figsize=(15,8))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()


# RSS Parser class
class Parser:
    def __init__(self, sources):
        self.sources = sources
        self.parsed_feed = []


    def parse(self):
        self.parsed_feed = []
        curr_count = 0

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
                if article_date is not None and ((datetime.datetime.now() - article_date) > datetime.timedelta(hours=12)):
                    continue  # skip old articles...

                # building the article 
                article = {
                    'title' : e['title'] if ('title' in e) else "",
                    'author': e['author'] if ('author' in e) else "",
                    'description' : soup.text if soup is not None else "",
                    'datetime' : article_date.isoformat() if article_date is not None else None,
                    'img' : imgurl['src'] if (imgurl is not None and imgurl.has_attr('src')) else "",
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

            print("{} loaded! ({} entries)".format(source['name'], len(self.parsed_feed) - curr_count))
            curr_count = len(self.parsed_feed)

        print("whole newsfeed loaded ({} entries)".format(len(self.parsed_feed)))


    def training_samples(self, num_articles=50):
        return random.sample(self.parsed_feed, num_articles)


    def sorted_feed(self):
        return sorted(self.parsed_feed, key=lambda x: try_parse_date(x['datetime']), reverse=True)


# NewsFeed class: (useless)
class NewsFeed:
    pass


class Miner:
    custom_sw = set(line.strip() for line in open('config/stopwords-it.txt'))
    stopwords = set(stopwords.words('italian')).union(custom_sw)

    ignore = lambda x: x # dumb function to ignore function handler that are not needed

    stem = {
        'porter': nltk.stem.PorterStemmer,
        'snowball': ItalianStemmer
    }

    vect = {
        'count': CountVectorizer,
        'tfidf': TfidfVectorizer
    }

    clf = {
        'mnb': MultinomialNB,
        'sgd': SGDClassifier,
        'svc': LinearSVC,
        'random_forest': RandomForestClassifier,
        'log_reg': LogisticRegression,
        'tree': DecisionTreeClassifier,
        'ada': AdaBoostClassifier
    }


    def __init__(self, dataset=None):
        self.dataset = pd.DataFrame(dataset)


########## TOKENIZATION #############

    @classmethod
    def word_tokenize(cls, corpus, stemmer='snowball'):
        # preparing stuff
        st = cls.stem[stemmer]()
        tokens = nlp(corpus)

        # tokenization + stemming
        tokens = [st.stem(unidecode(t.norm_)) for t in tokens if not (t.is_punct or t.is_space or t.like_num)
                                                                 and unidecode(t.norm_) not in cls.stopwords
                                                                 and len(t.text) >= 2]
        return tokens


    @classmethod
    def tokenize_article(cls, article,
                         should_remove_duplicates=False):

        corpus = article['title'] + '\n' + article['description']
        tokens = cls.word_tokenize(corpus)
        return list(set(tokens)) if should_remove_duplicates else tokens


    @classmethod
    def tokenize_list(cls, articles, should_remove_duplicates=False):
        if type(articles) is list:
            articles = pd.DataFrame(articles)
        return [Miner.tokenize_article(a, should_remove_duplicates=should_remove_duplicates) for _,a in articles.iterrows()]


########## TOKENIZATION #############


########## CROSS-VALIDATION + META-CLASSIFICATION #############
    @classmethod
    def show_confusion_matrix(cls, mat, labels):
        n_class = len(labels)
        df_cm = pd.DataFrame(
            mat, 
            index=labels,
            columns=labels
        )

        plt.figure(figsize = (n_class,n_class))
        sn.heatmap(df_cm, annot=True,  fmt='g')
        plt.show()


    @classmethod
    def cross_validate_confmat(cls, model, dataset, labels, n_class=None, folds=10):
        kf = StratifiedKFold(random_state=42, n_splits=folds, shuffle=True)
        total = 0
        totalMat = np.zeros((n_class,n_class))
        n_class = len(np.unique(labels)) if n_class is None else n_class

        # metrics
        f1 = []
        acc = []
        prc = []
        rcl = []

        for train_index, test_index in kf.split(dataset,labels):
            X_train = [dataset[i] for i in train_index]
            X_test = [dataset[i] for i in test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            model.fit(X_train, y_train)
            result = model.predict(X_test)
            
            mat = confusion_matrix(y_test, result)
            f1.append(f1_score(y_test, result, average='weighted'))
            acc.append(accuracy_score(y_test, result))
            prc.append(precision_score(y_test, result, average='weighted'))
            rcl.append(recall_score(y_test, result, average='weighted'))

            totalMat = totalMat + mat
            total = total + sum(y_test==result)
        
        print("---\n")
        print("F1 Scores: {}".format(mean(f1)))
        print("Accuracy Scores: {}".format(mean(acc)))
        print("Precision Scores: {}".format(mean(prc)))
        print("Recall Scores: {}".format(mean(rcl)))
        print("---\n")

        return (totalMat, total/len(labels))


    @classmethod
    def cross_validate_score(cls, model, dataset, labels, n_class=None, folds=10):
        n_class = len(np.unique(labels)) if n_class is None else n_class
        scores = cross_val_score(model, dataset, labels, cv=StratifiedKFold(random_state=42, n_splits=folds, shuffle=True))
        return scores

    
    @classmethod
    def cross_validate(cls, pl, ds, labels, n_class, show_mat=False):
        if show_mat:
            score = Miner.cross_validate_confmat(pl, ds, labels, n_class=n_class)
            Miner.show_confusion_matrix(score[0], get_categories())
        else:
            score = Miner.cross_validate_score(pl, ds, labels, n_class=n_class)
            print(score, mean(score))


    def meta_classify(self, show_mat=True, tuning=False, use_w2v=False):
        # preparing the trainingset
        ds =  Miner.tokenize_list(self.dataset)

        # preparing the target
        labels = self.dataset['tag'].to_numpy()
        n_class = len(get_categories())

        # preparing the vectorizer NOTE: NON FUNZIONA
        if use_w2v:
            # i'm going to use D2Vector
            vect = D2VTransformer( # some tuning is needed here (iguess)
                dm=1, 
                size=500, #maybeitstoomuch
                min_count=3, 
                iter=10, 
                seed=42
            ) 
        else: 
            # traditional TF-IDF
            vect = Miner.vect['tfidf'](
                #best parameters:
                max_features=14500,
                sublinear_tf=True,
                max_df=0.5,
                norm='l2',

                # init
                tokenizer=Miner.ignore, 
                preprocessor=Miner.ignore, 
                token_pattern=None,
            )
        
        # classifiers initialization
        classifiers = [
            # ('dt', Miner.clf['tree']()),
            # ('mnb', Miner.clf['mnb']()),
            ('svc', Miner.clf['svc'](C=0.41, random_state=42)),
            # ('rf', Miner.clf['random_forest'](random_state=42, n_estimators=10)),
            # ('Logistic Regression', Miner.clf['log_reg'](solver='lbfgs', multi_class='auto', random_state=42)),
            # ('ada', Miner.clf['ada'](n_estimators=10))
        ]


        # params tuning
        params = { 
            'rf' : {
                'clf__n_estimators': [2000],
                'clf__bootstrap': [True],
                'clf__max_depth': [None],
                'clf__max_features': ['sqrt'],
                'clf__min_samples_leaf': [2, 4],
                'clf__min_samples_split': [5, 10]
            },
            'mnb' : {},
            'svc' : {
                # 'vect__norm': [None, 'l1', 'l2']
                # 'vect__max_df': (0.5, 0.55, 0.65, 0.70, 0.75, 1.0),
                # 'vect__max_features' : [12000, 14500, 15000], # best MF: 14500
                # 'clf__C': np.arange(0.01, 4, 0.05) # best C: 0.46
            },
            'ada' : {
                'clf__n_estimators': [2000],
                'clf__algorithm': ['SAMME']
            },
            'lr' : {},
            'dt' : {}
        }

        for c in classifiers:
            print('\n---------------------------\n')
            
            # building the model
            pl = Pipeline(
                [('vect', vect)] +
                # ([] if use_w2v else [('sel', SelectPercentile(chi2, percentile=45))]) +
                [('clf', c[1])]
            )


            if tuning:
                print("\nTuning {} Hyper-Parameters with RandomizedSearchCV: \n".format(c[0]))
                model = RandomizedSearchCV(pl, params[c[0]], iid=True,
                    scoring='accuracy', cv=StratifiedKFold(random_state=42, n_splits=10, shuffle=True),
                    verbose=1,
                    n_jobs=2
                )
                model.fit(ds, labels)
                print(model.best_score_, model.best_params_)
            else:
                print('\n Regular CV 10 folds for ' + c[0] + '\n')
                Miner.cross_validate(pl, ds, labels, n_class, show_mat=show_mat)
            
            print('\n---------------------------\n')


        # NOTE: NOT REALLY WORKING...
        # print("\n\nDoing some actual metaclassification:\n")
        # sclf = StackingCVClassifier(classifiers=[c[1] for c in classifiers], 
        #                             meta_classifier=LogisticRegression())
        # classifiers.append(('stacking', sclf))
        # for c in classifiers:
        #    pl = Pipeline([
        #        ('vect', vect),
        #        ('clf', c[1])
        #    ])

        #    scores = cross_val_score(
        #        pl, 
        #        ds, labels, 
        #        cv=10, scoring='accuracy'
        #    )
        #    print("Accuracy: %0.2f (+/- %0.2f) [%s]"  % (scores.mean(), scores.std(), c[0]))

########## CROSS-VALIDATION + META-CLASSIFICATION #############

        
#####################   MODEL BUILDING   ##########################
    # this is the actual classifier (for app)
    def build_news_classifier(self):
        # this function should provide a pipeline trained object 
        # that i use to fit with the features extracted from the miner
        model = Pipeline([
            ('vect', Miner.vect['tfidf'](max_features=14500, tokenizer=Miner.ignore, preprocessor=Miner.ignore, token_pattern=None)),
            ('clf', Miner.clf['svc']())
        ])

        # this is wrong (i think), maybe it's better to do a shuffled 80-20 thing just to avoid overfitting
        ds = Miner.tokenize_list(self.dataset)
        labels = self.dataset['tag'].to_numpy()

        model.fit(ds, labels)
        return model

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
                'tag__':1, 
                'like':1, 
                'dislike':1, 
                'read':1
            }
        ).sort([('tag',1)]))
        return results


    def find_likabilityset(self):
        articles = self.db['articles']
        results = list(articles.find({
                'tag': {'$exists':True},
                '$or': [{ 'like': True }, { 'dislike': True }, { 'read': True }]
            }, {
                'title':1, 
                'description':1, 
                'tag':1, 
                'tag__':1, 
                'like':1, 
                'dislike':1, 
                'read':1
            }
        ).sort([('tag',1)]))
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
            'training': self.find_trainingset,
            'training_likability': self.find_likabilityset
        }
        return feed_descriptors[descriptor]()


    def close(self):
        self.client.close()