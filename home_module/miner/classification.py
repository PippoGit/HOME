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



import preprocessing as pp

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


vectorizer = {
    'count': CountVectorizer,
    'tfidf': TfidfVectorizer
}

classifier = {
    'mnb': MultinomialNB,
    'sgd': SGDClassifier,
    'svc': LinearSVC,
    'random_forest': RandomForestClassifier,
    'log_reg': LogisticRegression,
    'tree': DecisionTreeClassifier,
    'ada': AdaBoostClassifier
}

news_categories = [
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


def dumb_function(x):
    return x


def labelize_likability(article):
    # likability = max(0.5 + (article['like'])*0.5 - (article['dislike'])*0.5 - (not article['read'])*0.2, 0)
    likability = 1 # deprecated 

    if article['dislike']:
        return ("DISLIKE", likability)
        
    if article['like']:
        return ("LIKE", likability)

    return ("READ", likability) # should not happen


########## CROSS-VALIDATION + META-CLASSIFICATION #############
def show_confusion_matrix(mat, labels):
    n_class = len(labels)
    df_cm = pd.DataFrame(
        mat, 
        index=labels,
        columns=labels
    )

    plt.figure(figsize = (n_class,n_class))
    sn.heatmap(df_cm, annot=True,  fmt='g')
    plt.show()


def cross_validate_fullscores(model, dataset, labels, n_class=None, folds=10):
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

    
def cross_validate(pl, ds, labels, n_class, show_mat=False, txt_labels=None):
    score = cross_validate_fullscores(pl, ds, labels, n_class=n_class)
    if show_mat:
        show_confusion_matrix(score[0], txt_labels)


def init_simple_classifiers(clf='nc'):
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
        ('dt', classifier['tree']()), # dt is so bad, probably should not even be considered
        ('mnb', classifier['mnb']()),
        ('svc', classifier['svc'](C=params[clf]['C'], random_state=params[clf]['random_state'])),
        ('Logistic Regression', classifier['log_reg'](solver=params[clf]['lr_solver'], multi_class=params[clf]['lr_multi_class'], random_state=params[clf]['random_state'])),
    ]


def init_ensmeta_classifiers(simple_classifier_list, clf='nc'):
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
        ("AdaBoost", classifier['ada'](n_estimators=params[clf]['ada_estimators'])),
        ("RandomForest", classifier['random_forest'](random_state=params[clf]['random_state'], n_estimators=params[clf]['rf_estimators'])),
        ("XGBClassifier", XGBClassifier(max_depth=params[clf]['xgb_max_depth'], n_estimators=params[clf]['xgb_estimators'], learning_rate=params[clf]['xgb_learning_rate'])),
        
        ("VotingClassifier", VotingClassifier(estimators=simple_classifier_list, voting=params[clf]['voting'])),
        ("BaggingClassifier", BaggingClassifier(base_estimator=classifier['svc'](C=params[clf]['C'], random_state=params[clf]['random_state']), 
                                                n_estimators=params[clf]['bc_estimators'], 
                                                random_state=params[clf]['random_state']))
    ]


def test_stacking_classifier(classifiers, ds, labels):
    print("StackingClassifier: \n")
    # trying StackingClassifier (this is so bad it doesn't even worth it)
    sclf = StackingCVClassifier(classifiers=[c[1] for c in classifiers], 
                                meta_classifier=LogisticRegression(solver='lbfgs', multi_class='auto', random_state=42),
                                use_features_in_secondary=True)

    # building a list of Pipeline vect-classifier
    pipeline = Pipeline([
        ('vect', init_vectorizer()),
        ('denser', DenseTransformer()), # StackingCV is not working with Sparse matrix (maybe this is why it sucks so much)
        ('sclf', sclf)
    ])

    encoded_label = LabelEncoder().fit_transform(labels) # don't know why it doesn't work with string values

    # trying to cross_validate the stack...
    scores = cross_val_score(pipeline, ds, encoded_label, 
                                cv=10, scoring='accuracy')
    print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std())) # actually it is really bad, like 30%


def init_vectorizer():
    return vectorizer['tfidf'](
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


def build_lc_model(model):
    return Pipeline([
            ('union', FeatureUnion([
                    # Pipeline for article's content
                    ('content', Pipeline([
                        ('selector', ItemSelector(key='content')),
                        ('vect', init_vectorizer()),
                    ])),
                    # Pipeline for article's tag
                    ('tag', Pipeline([
                        ('selector', ItemSelector(key='tag')),
                        ('enc', ModifiedLabelEncoder())
                    ]))
                ])),
            model
        ])


def build_nc_model(model):
    return Pipeline([
            ('vect', init_vectorizer()),
            # ('sel', SelectPercentile(chi2, percentile=45)), # not so useful...
            ('clf', model[1])
    ])


def meta_classify_nc(dataset, show_mat=False, tuning=False):
    # preparing the trainingset
    ds = pp.tokenize_list(dataset)

    # preparing the targets
    labels = dataset['tag'].to_numpy()
    n_class = len(news_categories)
    
    # classifiers initialization
    classifiers = init_simple_classifiers('nc')

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
        pl = build_nc_model(c)

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
            cross_validate(pl, ds, labels, n_class, show_mat=show_mat, txt_labels=news_categories)
        
        print('\n---------------------------\n')



    print("\n\nEnsembles and Meta-Classifiers:\n")
    
    # set the seed for some classifiers...
    np.random.seed(42)
    
    # init the list...
    ens_meta_classifiers = init_ensmeta_classifiers(classifiers, 'nc')

    for c in ens_meta_classifiers:
        print("\nCV 10 folds - " + c[0])
        # building the pipeline vectorizer-classifier
        pipeline = build_nc_model(c)

        # Cross_validating the model
        cross_validate(pipeline, ds, labels, n_class, show_mat=show_mat, txt_labels=news_categories)



def meta_classify_lc(dataset, show_mat=False, tuning=False):
    # preparing the inputs
    ds = pd.DataFrame()
    ds['content'] = pp.tokenize_list(dataset)
    ds['tag'] = dataset['tag']

    # preparing the targets
    labels = np.asarray([labelize_likability(a)[0] for _,a in dataset.iterrows()])
    
    # classifiers initialization
    classifiers = init_simple_classifiers('lc')

    print("\n\nSimple Classifiers:\n")
    for c in classifiers:
        print('\n---------------------------\n')
        
        # building the model
        pl = build_lc_model(c)

        print('\n Regular CV 10 folds for ' + c[0] + '\n')
        cross_validate(pl, ds, labels, 2, show_mat=show_mat, txt_labels=['DISLIKED', 'LIKED'])
        print('\n---------------------------\n')

    print("\n\nEnsembles and Meta-Classifiers:\n")
    np.random.seed(42)

    ens_meta_classifiers = init_ensmeta_classifiers(classifiers, 'lc')

    for c in ens_meta_classifiers:
        print("CV 10 folds - " + c[0])

        # building the pipeline vectorizer-classifier
        pl = build_lc_model(c)
        
        # Cross_validating the model (dunno y its not working with the )
        cross_validate(pl, ds, labels, 2, show_mat=show_mat, txt_labels=['DISLIKED', 'LIKED'])

    # trying StackingClassifier (this is so bad it doesn't even worth it)
    # Miner.test_stacking_classifier(classifiers, ds, labels)


########## CROSS-VALIDATION + META-CLASSIFICATION #############

    
#####################   MODEL DEPLOY   ##########################
def deploy_news_classifier(dataset):
    # this function should provide a pipeline trained object 
    # that i use to fit with the features extracted from the miner

    clf = classifier['svc'](
        C=0.51
    )

    model = build_nc_model(('clf', clf))

    ds = pp.tokenize_list(dataset)
    labels = dataset['tag'].to_numpy()

    cross_validate_fullscores(model, ds, labels, n_class=9)        
    joblib.dump(model, 'model/nws_clf.pkl')
    return model


def deploy_likability_predictor(dataset):
    # preparing the inputs
    ds = pd.DataFrame()
    ds['content'] = pp.tokenize_list(dataset)
    ds['tag'] = dataset['tag']

    # preparing the targets
    labels = np.asarray([labelize_likability(a)[0] for _,a in dataset.iterrows()])

    # building the model...
    clf = classifier['svc']()
    model = build_lc_model(('clf', clf))

    cross_validate_fullscores(model, ds, labels, n_class=2)        
    joblib.dump(model, 'model/lik_prd.pkl')
    return model


def load_likability_predictor():
    model = joblib.load('model/lik_prd.pkl')
    return model


def load_news_classifier():
    model = joblib.load('model/nws_clf.pkl')
    return model

#####################   MODEL BUILDING   ##########################