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
from statistics import mean, stdev
from sklearn.externals import joblib

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
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, learning_curve
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, f1_score, recall_score, precision_score


from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from mlxtend.classifier import StackingCVClassifier
from mlxtend.preprocessing import DenseTransformer

from xgboost import XGBClassifier

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.decomposition import TruncatedSVD

# SPACY and Stuff...
import spacy
nlp = spacy.load('it_core_news_sm')


def test(skip_parse=False, meta_classify=False, show_mat=True, tuning=False, wordcloud=False):
    # importing configuration 
    print("\nimporting config file...") 
    config = load_config()

    # preparing the components
    print("\npreparing the components...\n")
    db = DBConnector(**config['db'])
    feed_parser = Parser(config['feeds']) 

    # loading initial feed
    print("\nloading feeds...") 
    if skip_parse:
        print("actually i'm going to skip this (training purposes)\n")
    else:
        feed_parser.parse()

    # filtering the dataset using some machinelearning magic...
    print("preparing dataset for the miner...\n")
    miner = Miner()
    
    if meta_classify:
        print('\nmeta-classifing NC ... ')
        miner.meta_classify_nc(dataset=pd.DataFrame(db.find_trainingset()), show_mat=show_mat, tuning=tuning)
        print('\nmeta-classifing LC ... ')
        miner.meta_classify_lc(pd.DataFrame(db.find_likabilityset()), tuning=tuning, show_mat=show_mat)

    if wordcloud:
        show_wordcloud(flatten(Miner.tokenize_list(db.find_trainingset())))

    print('done!\nhome is ready! \n\tdict: {"config", "db", "feed_parser", "newsfeed", "miner"}\n')
    return {'config': config, 'db': db, 'feed_parser': feed_parser, 'miner': miner}


def test_classifiers(nc=True, lc=True, show_mat=False, tuning=False):
     # importing configuration 
    print("\nimporting config file...") 
    config = load_config()

    # preparing the components
    print("\npreparing the components...\n")
    db = DBConnector(**config['db'])
    miner = Miner()

    if nc:
        print('\nmeta-classifing NC ... ')
        miner.meta_classify_nc(dataset=pd.DataFrame(db.find_trainingset()), show_mat=show_mat, tuning=tuning)
    
    if lc:
        print('\nmeta-classifing LC ... ')
        miner.meta_classify_lc(dataset=pd.DataFrame(db.find_likabilityset()), tuning=tuning, show_mat=show_mat)


def deploy_models():
    # importing configuration 
    print("\nimporting config file...") 
    config = load_config()

    # preparing the components
    print("\npreparing the components...\n")
    db = DBConnector(**config['db'])

    print("building the news classifier...")
    Miner.deploy_news_classifier(pd.DataFrame(db.find_trainingset()))
    print("building the likability predictor...")
    Miner.deploy_likability_predictor(pd.DataFrame(db.find_likabilityset()))
    print("models built!")


# some util function 
def load_config():
    with open("config/config.json") as f:
        config = json.load(f)
    return config


def likability(article):
    # likability = max(0.5 + (article['like'])*0.5 - (article['dislike'])*0.5 - (not article['read'])*0.2, 0)
    likability = 1 # deprecated 

    if article['dislike']:
        return ("DISLIKE", likability)
        
    if article['like']:
        return ("LIKE", likability)

    return ("READ", likability) # should not happen


def list_union(lst1, lst2): 
    final_list = list(set(lst1) | set(lst2)) 
    return final_list 


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


class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        return data[self.key].to_numpy()

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class ModifiedLabelEncoder(LabelEncoder):
    def fit_transform(self, y, *args, **kwargs):
        return super().fit_transform(y).reshape(-1, 1)

    def transform(self, y, *args, **kwargs):
        return super().transform(y).reshape(-1, 1)


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


# NewsFeed class
class NewsFeed:
    def __init__(self, nws_clf, lik_prd):
        self.nws_clf = nws_clf
        self.lik_prd = lik_prd
        self.feed = None


    def to_list(self):
        return self.feed.to_dict('records')


    def build_feed(self, parsed_feed):
        self.feed = pd.DataFrame(parsed_feed)

        features = pd.DataFrame()        
        features['content']     = Miner.tokenize_list(parsed_feed)
        features['tag']         = self.nws_clf.predict(features['content'])

        self.feed['likability'] = self.lik_prd.predict(features)
        self.feed['predicted_tag'] = features['tag']

        self.feed = self.feed[self.feed['likability'] == 'LIKE'].drop('likability', axis=1)



def dumb_function(x):
    return x

class Miner:
    custom_sw = set(line.strip() for line in open('config/stopwords-it.txt'))
    stopwords = set(stopwords.words('italian')).union(custom_sw)

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


########## TOKENIZATION #############

    @classmethod
    def word_tokenize(cls, corpus, stemmer='snowball'):
        # preparing stuff
        st = cls.stem[stemmer]()
        tokens = nlp(corpus) # this is really slow, but can't avoid using it!

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
    def tokenize_likability(cls, article):
        corpus = article['title']
        return cls.word_tokenize(corpus)
        

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
    def cross_validate_fullscores(cls, model, dataset, labels, n_class=None, folds=10):
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
            if isinstance(dataset, pd.DataFrame):
                X_train = dataset.iloc[train_index]
                X_test = dataset.iloc[test_index]
            else:
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
        print("F1 Scores: %0.4f [ +/- %0.4f]" % (mean(f1), stdev(f1)))
        print("Accuracy Scores: %0.4f [ +/- %0.4f]" % (mean(acc), stdev(acc)))
        print("Precision Scores: %0.4f [ +/- %0.4f]" % (mean(prc), stdev(prc)))
        print("Recall Scores: %0.4f [ +/- %0.4f]" % (mean(rcl), stdev(rcl)))
        print("---\n")
        print(totalMat)
        return (totalMat, total/len(labels))

    
    @classmethod
    def cross_validate(cls, pl, ds, labels, n_class, show_mat=False, txt_labels=None):
        score = cls.cross_validate_fullscores(pl, ds, labels, n_class=n_class)
        if show_mat:
            cls.show_confusion_matrix(score[0], txt_labels)


    @classmethod
    def init_simple_classifiers(cls, clf='nc'):
        params = {
            'nc': {
                'C':0.51,
                'random_state': 42,
                'lr_solver': 'lbfgs',
                'lr_multi_class':'auto',
            },
            'lc' : {
                'C': 1,
                'random_state': 42,
                'lr_solver': 'lbfgs',
                'lr_multi_class':'auto',
            }
        }

        return [
            ('dt', cls.clf['tree']()), # dt is so bad, probably should not even be considered
            ('mnb', cls.clf['mnb']()),
            ('svc', cls.clf['svc'](C=params[clf]['C'], random_state=params[clf]['random_state'])),
            ('Logistic Regression', cls.clf['log_reg'](solver=params[clf]['lr_solver'], multi_class=params[clf]['lr_multi_class'], random_state=params[clf]['random_state'])),
        ]


    @classmethod
    def init_ensmeta_classifiers(cls, simple_classifier_list, clf='nc'):
        params = {
            'nc': {
                'C':0.51,
                'random_state': 42,
                'lr_solver': 'lbfgs',
                'lr_multi_class':'auto',
                'bc_estimators': 100,
                'voting': 'hard',
                'ada_estimators': 100,
                'rf_estimators': 100,
                'xgb_estimators': 100,
                'xgb_max_depth': 3,
                'xgb_learning_rate': 0.1,
            },
            'lc' : {
                'C': 1,
                'random_state': 42,
                'lr_solver': 'lbfgs',
                'lr_multi_class':'auto',
                'bc_estimators': 100,
                'voting': 'hard',
                'ada_estimators': 100,
                'rf_estimators': 100,
                'xgb_estimators': 100,
                'xgb_max_depth': 3,
                'xgb_learning_rate': 0.1,
            }
        }

        return [
            ("AdaBoost", cls.clf['ada'](n_estimators=params[clf]['ada_estimators'])),
            ("RandomForest", cls.clf['random_forest'](random_state=params[clf]['random_state'], n_estimators=params[clf]['rf_estimators'])),
            ("XGBClassifier", XGBClassifier(max_depth=params[clf]['xgb_max_depth'], n_estimators=params[clf]['xgb_estimators'], learning_rate=params[clf]['xgb_learning_rate'])),
         
            ("VotingClassifier", VotingClassifier(estimators=simple_classifier_list, voting=params[clf]['voting'])),
            ("BaggingClassifier", BaggingClassifier(base_estimator=cls.clf['svc'](C=params[clf]['C'], random_state=params[clf]['random_state']), 
                                                    n_estimators=params[clf]['bc_estimators'], 
                                                    random_state=params[clf]['random_state']))
        ]


    @classmethod
    def test_stacking_classifier(cls, classifiers, ds, labels):
        print("StackingClassifier: \n")
        # trying StackingClassifier (this is so bad it doesn't even worth it)
        sclf = StackingCVClassifier(classifiers=[c[1] for c in classifiers], 
                                    meta_classifier=LogisticRegression(solver='lbfgs', multi_class='auto', random_state=42),
                                    use_features_in_secondary=True)

        # building a list of Pipeline vect-classifier
        pipeline = Pipeline([
            ('vect', cls.init_vectorizer()),
            ('denser', DenseTransformer()), # StackingCV is not working with Sparse matrix (maybe this is why it sucks so much)
            ('sclf', sclf)
        ])

        encoded_label = LabelEncoder().fit_transform(labels) # don't know why it doesn't work with string values

        # trying to cross_validate the stack...
        scores = cross_val_score(pipeline, ds, encoded_label, 
                                 cv=10, scoring='accuracy')
        print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std())) # actually it is really bad, like 30%


    @classmethod
    def init_vectorizer(cls):
        return cls.vect['tfidf'](
            #best parameters:
            max_features=14500,
            sublinear_tf=True,
            min_df=1,
            max_df=0.5,
            norm='l2',
            ngram_range=(1, 1),

            # init
            tokenizer=dumb_function, 
            preprocessor=dumb_function, 
            token_pattern=None,
        )


    @classmethod
    def build_lc_model(cls, model):
        return Pipeline([
                ('union', FeatureUnion([
                        # Pipeline for article's content
                        ('content', Pipeline([
                            ('selector', ItemSelector(key='content')),
                            ('vect', cls.init_vectorizer()),
                        ])),
                        # Pipeline for article's tag
                        ('tag', Pipeline([
                            ('selector', ItemSelector(key='tag')),
                            ('enc', ModifiedLabelEncoder())
                        ]))
                    ])),
                model
            ])


    @classmethod
    def build_nc_model(cls, model):
        return Pipeline([
                ('vect', cls.init_vectorizer()),
                # ('sel', SelectPercentile(chi2, percentile=45)), # not so useful...
                ('clf', model[1])
        ])


    def meta_classify_nc(self, dataset, show_mat=False, tuning=False):
        # preparing the trainingset
        ds =  Miner.tokenize_list(dataset)

        # preparing the targets
        labels = dataset['tag'].to_numpy()
        n_class = len(get_categories())
        
        # classifiers initialization
        classifiers = Miner.init_simple_classifiers('nc')

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
                # 'vect__ngram_range': [(1, 1), (1, 2), (1, 3), (2, 2)],
                # 'vect__min_df': (1, 0.01, 0.025),
                # 'vect__norm': [None, 'l1', 'l2']
                # 'vect__max_df': (0.5, 0.55, 0.65, 0.70, 0.75, 1.0),
                # 'vect__max_features' : [12000, 14500, 15000], # best MF: 14500
                # 'clf__C': [1, 10, 20, 25, 50, 75, 100] # best C: 0.51
            },
            'ada' : {
                'clf__n_estimators': [2000],
                'clf__algorithm': ['SAMME']
            },
            'lr' : {},
            'dt' : {}
        }

        print("\n\nSimple Classifiers:\n")
        for c in classifiers:
            print('\n---------------------------\n')
            # building the model
            pl = Miner.build_nc_model(c)

            if tuning:
                print("\nTuning {} Hyper-Parameters with GridSearchCV: \n".format(c[0]))
                model = GridSearchCV(pl, params[c[0]], iid=True,
                    scoring='accuracy', cv=StratifiedKFold(random_state=42, n_splits=10, shuffle=True),
                    verbose=1,
                    n_jobs=2
                )
                model.fit(ds, labels)
                print(model.best_score_, model.best_params_)
            else:
                print('\n Regular CV 10 folds for ' + c[0] + '\n')
                Miner.cross_validate(pl, ds, labels, n_class, show_mat=show_mat, txt_labels=get_categories())
            
            print('\n---------------------------\n')



        print("\n\nEnsembles and Meta-Classifiers:\n")
        
        # set the seed for some classifiers...
        np.random.seed(42)
        
        # init the list...
        ens_meta_classifiers = Miner.init_ensmeta_classifiers(classifiers, 'nc')

        for c in ens_meta_classifiers:
            print("\nCV 10 folds - " + c[0])
            # building the pipeline vectorizer-classifier
            pipeline = Miner.build_nc_model(c)

            # Cross_validating the model
            Miner.cross_validate(pipeline, ds, labels, n_class, show_mat=show_mat, txt_labels=get_categories())

        # trying StackingClassifier (this is so bad it doesn't even worth it)
        # Miner.test_stacking_classifier(classifiers, ds, labels)


    def meta_classify_lc(self, dataset, show_mat=False, tuning=False):
        # preparing the inputs
        ds = pd.DataFrame()
        ds['content'] = Miner.tokenize_list(dataset)
        ds['tag'] = dataset['tag']

        # preparing the targets
        labels = np.asarray([likability(a)[0] for _,a in dataset.iterrows()])
        
        # classifiers initialization
        classifiers = Miner.init_simple_classifiers('lc')

        print("\n\nSimple Classifiers:\n")
        for c in classifiers:
            print('\n---------------------------\n')
            
            # building the model
            pl = Miner.build_lc_model(c)

            print('\n Regular CV 10 folds for ' + c[0] + '\n')
            Miner.cross_validate(pl, ds, labels, 2, show_mat=show_mat, txt_labels=['DISLIKED', 'LIKED'])
            print('\n---------------------------\n')

        print("\n\nEnsembles and Meta-Classifiers:\n")
        np.random.seed(42)

        ens_meta_classifiers = Miner.init_ensmeta_classifiers(classifiers, 'lc')

        for c in ens_meta_classifiers:
            print("CV 10 folds - " + c[0])

            # building the pipeline vectorizer-classifier
            pl = Miner.build_lc_model(c)
            
            # Cross_validating the model (dunno y its not working with the )
            Miner.cross_validate(pl, ds, labels, 2, show_mat=show_mat, txt_labels=['DISLIKED', 'LIKED'])

        # trying StackingClassifier (this is so bad it doesn't even worth it)
        # Miner.test_stacking_classifier(classifiers, ds, labels)


########## CROSS-VALIDATION + META-CLASSIFICATION #############

    
#####################   MODEL DEPLOY   ##########################
    @classmethod
    def deploy_news_classifier(cls, dataset):
        # this function should provide a pipeline trained object 
        # that i use to fit with the features extracted from the miner

        clf = Miner.clf['svc'](
            C=0.51
        )

        model = Miner.build_nc_model(('clf', clf))

        ds = Miner.tokenize_list(dataset)
        labels = dataset['tag'].to_numpy()

        Miner.cross_validate_fullscores(model, ds, labels, n_class=9)        
        joblib.dump(model, 'model/nws_clf.pkl')
        return model


    @classmethod
    def deploy_likability_predictor(cls, dataset):
        # preparing the inputs
        ds = pd.DataFrame()
        ds['content'] = Miner.tokenize_list(dataset)
        ds['tag'] = dataset['tag']

        # preparing the targets
        labels = np.asarray([likability(a)[0] for _,a in dataset.iterrows()])

        # building the model...
        clf = Miner.clf['svc']()
        model = Miner.build_lc_model(('clf', clf))

        Miner.cross_validate_fullscores(model, ds, labels, n_class=2)        
        joblib.dump(model, 'model/lik_prd.pkl')
        return model


    @classmethod
    def load_likability_predictor(cls):
        model = joblib.load('model/lik_prd.pkl')
        return model

    
    @classmethod
    def load_news_classifier(cls):
        model = joblib.load('model/nws_clf.pkl')
        return model

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
                '$or': [{ 'like': True }, { 'dislike': True }]
            }, {
                'title':1, 
                'description':1, 
                'tag':1, 
                'source': 1,
                'tag__':1,
                'like':1, 
                'dislike':1, 
                'read':1
            }
        ).sort([('dislike',1)]))
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
            'num_likes': self.db.articles.count({'like':True}),
            'num_dislikes': self.db.articles.count({'dislike':True}),
            'num_read': self.db.articles.count({'read':True}),
            'num_ignored': self.db.articles.count({'like':False, 'dislike':False, 'read':False}),
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
