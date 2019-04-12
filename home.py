# parsing & db stuff
import feedparser
from bs4 import BeautifulSoup
import pymongo

# util
import pandas as pd
import numpy as np
import hashlib, datetime, ssl, random, json, re, string
from collections import defaultdict

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# natural language text processing
import nltk
from nltk.corpus import stopwords

# machine learning
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder


from gensim.models import Word2Vec # NOTE: NON FUNZIONA
from gensim.sklearn_api import D2VTransformer # NOTE: NON FUNZIONA

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold 
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


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



def test(skip_parse=False, meta_classify=False, use_w2v=False):

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
        miner.meta_classify(use_w2v=use_w2v)



    from collections import Counter
    l = Miner.tokenize_list(miner.dataset)
    flatten = [e for sl in l for e in sl]

    should_not_happen = [e for e in flatten if e in Miner.stopwords]
    print(len(should_not_happen))

    # WORLDCLOUDSTUFF...
    # word_cloud_dict = Counter(flatten)
    # wordcloud = WordCloud(width = 1000, height = 500).generate_from_frequencies(word_cloud_dict)

    # plt.figure(figsize=(15,8))
    # plt.imshow(wordcloud)
    # plt.axis("off")
    # plt.show()

    # plt.savefig('yourfile.png', bbox_inches='tight')
    # plt.close()


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


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt



# RSS Parser class
class Parser:
    def __init__(self, sources):
        self.sources = sources
        self.parsed_feed = []


    def parse(self):
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
                if article_date is not None and ((datetime.datetime.now() - article_date) > datetime.timedelta(hours=24)):
                    continue  # skip old articles...

                # building the article
                article = {
                    'title' : e['title'] if ('title' in e) else "",
                    'author': e['author'] if ('author' in e) else "",
                    'description' : soup.text if soup is not None else "",
                    'datetime' : article_date.isoformat() if article_date is not None else None,
                    'img' : imgurl['src'] if (imgurl is not None and 'src' in imgurl) else "",
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


# NewsFeed class: (useless)
class NewsFeed:
    pass


class Miner:
    custom_sw = set(line.strip() for line in open('config/stopwords-it.txt'))
    stopwords = set(stopwords.words('italian')).union(custom_sw)

    max_features = 8000 # 5555 
    ignore = lambda x: x # dumb function to ignore function handler that are not needed

    stemmers = {
        'porter': nltk.stem.PorterStemmer,
        'snowball': nltk.stem.SnowballStemmer
    }

    vectorizers = {
        'count': CountVectorizer,
        'tfidf': TfidfVectorizer,
        'w2v': D2VTransformer # NOTE NON FUNZIONA!!!!!
    }

    classifiers = {
        'multinomial_nb': MultinomialNB,
        'sgd': SGDClassifier,
        'linear_svc': LinearSVC,
        'random_forest': RandomForestClassifier,
        'logistic_regression': LogisticRegression,
        'tree': DecisionTreeClassifier
    }


    def __init__(self, dataset=None):
        self.dataset = pd.DataFrame(dataset)


########## TOKENIZATION #############
    @classmethod
    def remove_stopwords(cls, tokens):
        # print("Removing stopwords...")
        rmvd =  [w for w in tokens if w not in cls.stopwords]
        return rmvd


    @classmethod
    def word_tokenize(cls, text, ignore_stopwords=False, clean=True):
        
        if clean:
            # replace symbols and multiple spaces with a single space
            text = re.sub('( +)|(' + r'\W+' + ')', ' ', text)            

            # remove numbers that are not part of a word            
            text = re.sub(r"\b\d+\b", "", text)

        table = str.maketrans('', '', string.punctuation)
        tokens = [t.lower().translate(table) for t in nltk.word_tokenize(text)]
        tokens = cls.remove_stopwords(tokens) if ignore_stopwords else tokens

        return tokens


    @classmethod
    def build_article_tokens(cls, article, remove_duplicates=False, ignore_stopwords=False, clean=True):
        t = dict()
        t['title'] = cls.word_tokenize(article['title'], ignore_stopwords, clean)
        t['description'] = cls.word_tokenize(article['description'], ignore_stopwords, clean)

        t_td = t['title'] + t['description']

        return list(set(t_td)) if remove_duplicates else t_td


    @classmethod
    def clean_article_tokens(cls, token):
        regex = re.compile(r'\w+') # remove punctuation
        return [x for x in token if regex.match(x)]


    @classmethod
    def stem_article_tokens(cls, token, stemmer):
        return [stemmer.stem(w) for w in token]


    @classmethod
    def tokenize_article(cls, article, stemmer='snowball',
                         should_remove_duplicates=False, should_ignore_sw=True, should_clean=True, should_stem=True):
        # tokenize article
        tokens = cls.build_article_tokens(article, 
                                            remove_duplicates=should_remove_duplicates, 
                                            ignore_stopwords=should_ignore_sw, 
                                            clean=should_clean)

        return cls.stem_article_tokens(tokens, cls.stemmers[stemmer]('italian')) if should_stem else tokens


    @classmethod
    def tokenize_list(cls, articles):
        if type(articles) is list:
            articles = pd.DataFrame(articles)
        return [Miner.tokenize_article(a) for _,a in articles.iterrows()]

########## TOKENIZATION #############


########## GloVe and Word2Vec ########### (KINDA-USELESS)
    @classmethod
    def build_w2v(cls, wordemb_path='w2v/glove_WIKI'):
        return Word2Vec(wordemb_path) # load CNR Glove Model (PRETRAINED)
########## GloVe and Word2Vec ###########


########## CROSS-VALIDATION + META-CLASSIFICATION #############
    @classmethod
    def cross_validate_pipeline(cls, pipeline, dataset, labels, n_class=2, folds=10):
        kf = StratifiedKFold(n_splits=folds)
        total = 0
        totalMat = np.zeros((n_class,n_class))

        for train_index, test_index in kf.split(dataset,labels):
            X_train = [dataset[i] for i in train_index]
            X_test = [dataset[i] for i in test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            pipeline.fit(X_train, y_train)
            result = pipeline.predict(X_test)
            
            mat = confusion_matrix(y_test, result)

            totalMat = totalMat + mat
            total = total + sum(y_test==result)
            
        return (totalMat, total/len(labels))


    @classmethod
    def cross_validate(cls, dataset, labels, classifier, vectorizer=None, n_class=2, folds=10):
        kf = StratifiedKFold(n_splits=folds, random_state=42, shuffle=True)
        total = 0
        totalMat = np.zeros((n_class,n_class))
        
        for train_index, test_index in kf.split(dataset,labels):
            X_train = [dataset[i] for i in train_index]
            X_test = [dataset[i] for i in test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            train_features = vectorizer.fit_transform(X_train) # if vectorizer else X_train 
            test_features = vectorizer.transform(X_test) # if vectorizer else X_test 

            classifier.fit(train_features,y_train)
            result = classifier.predict(test_features)
            
            mat = confusion_matrix(y_test, result)

            totalMat = totalMat + mat
            total = total + sum(y_test==result)
            
        return (totalMat, total/len(labels))


    def meta_classify(self, use_w2v=False):
        print('\nTokenizing the articles in the dataset...')
        ds = Miner.tokenize_list(self.dataset)

        # preparing the labels...
        labels = self.dataset['tag'].to_numpy()

        # preparing modules of the classifier
        print('Preparing the modules (vectorizer and list of classifiers)\n')
        
        # vectorizer (if use_w2v the vectorizer will be None)
        # vect = Miner.vectorizers['w2v'](dm=0, size=300, negative=5, hs=0, min_count=2, sample = 0, workers=2) if use_w2v else Miner.vectorizers['tfidf'](max_features=Miner.max_features, tokenizer=Miner.ignore, preprocessor=Miner.ignore, token_pattern=None)

        vect = Miner.vectorizers['tfidf'](max_features=Miner.max_features, tokenizer=Miner.ignore, preprocessor=Miner.ignore, token_pattern=None)
        
        # classifiers' list 
        classifiers = [
            ('Decision Tree Classifier', Miner.classifiers['tree']()),
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
            pl = Pipeline([
                ('vect', vect),
                ('clf', c[1])
            ])
            score = Miner.cross_validate_pipeline(pl, ds, labels, n_class=len(get_categories()))

            print(score)

            fig = plt.figure()
            ax = fig.add_subplot(111)
            cax = ax.matshow(score[0])
            plt.title('Confusion matrix of ' + c[0])
            fig.colorbar(cax)
            ax.set_xticklabels([''] + get_categories())
            ax.set_yticklabels([''] + get_categories())
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.show()

        print('\n---------------------------')
        print('\nMeta-Classification Done!\n')
        return
########## CROSS-VALIDATION + META-CLASSIFICATION #############

        
#####################   MODEL BUILDING   ##########################
    # this is the actual classifier (for app)
    def build_news_classifier(self):
        # this function should provide a pipeline trained object 
        # that i use to fit with the features extracted from the miner
        model = Pipeline([
            ('vect', Miner.vectorizers['tfidf'](max_features=Miner.max_features, tokenizer=Miner.ignore, preprocessor=Miner.ignore, token_pattern=None)),
            ('clf', Miner.classifiers['linear_svc']())
        ])

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
            'training': self.find_trainingset
        }
        return feed_descriptors[descriptor]()


    def close(self):
        self.client.close()