# util
import pandas as pd
import numpy as np
from statistics import mean, stdev
from sklearn.externals import joblib
from sklearn.utils import shuffle, indexable
import itertools
import pickle
from scipy import stats

# plot
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sn
 
# machine learning
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder, LabelBinarizer

from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve, RepeatedStratifiedKFold, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, get_scorer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import SelectPercentile, chi2, SelectFromModel

from mlxtend.classifier import StackingCVClassifier
from mlxtend.preprocessing import DenseTransformer
from mlxtend.evaluate import paired_ttest_5x2cv, paired_ttest_kfold_cv, paired_ttest_resampled

from xgboost import XGBClassifier

from home.miner import preprocessing as pp

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


def custom_paired_ttest_cv(estimator1, estimator2, X, y,
                          cv=10,
                          scoring=None,
                          shuffle=False,
                          random_seed=None):
    
    kf = StratifiedKFold(random_state=random_seed, n_splits=cv, shuffle=True)  

    if scoring is None:
        if estimator1._estimator_type == 'classifier':
            scoring = 'accuracy'
        elif estimator1._estimator_type == 'regressor':
            scoring = 'r2'
        else:
            raise AttributeError('Estimator must '
                                 'be a Classifier or Regressor.')
    if isinstance(scoring, str):
        scorer = get_scorer(scoring)
    else:
        scorer = scoring

    score_diff = []

    # this is probably wrong :(
    for train_index, test_index in kf.split(X, y):
        ##### THIS IS WHERE IT BECOMES "CUSTOM"
        if isinstance(X, pd.DataFrame):
            X_train = X.iloc[train_index]
            X_test = X.iloc[test_index]
        else:
            X_train = [X[i] for i in train_index]
            X_test = [X[i] for i in test_index]
        #####

        y_train, y_test = y[train_index], y[test_index]

        estimator1.fit(X_train, y_train)
        estimator2.fit(X_train, y_train)

        est1_score = scorer(estimator1, X_test, y_test)
        est2_score = scorer(estimator2, X_test, y_test)
        score_diff.append(est1_score - est2_score)

    avg_diff = np.mean(score_diff)

    numerator = avg_diff * np.sqrt(cv)
    denominator = np.sqrt(sum([(diff - avg_diff)**2 for diff in score_diff])
                          / (cv - 1))
    t_stat = numerator / denominator

    pvalue = stats.t.sf(np.abs(t_stat), cv - 1)*2.
    return float(t_stat), float(pvalue)


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


def plot_learning_curve(estimator, title, X, y, ylim=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5), random_state=42):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=10, shuffle=True, random_state=random_state, n_jobs=n_jobs, train_sizes=train_sizes)
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
    plt.show()
    return plt


def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)



########## CROSS-VALIDATION + META-CLASSIFICATION #############
def plot_confusion_matrix(mat, labels):
    n_class = len(labels)
    df_cm = pd.DataFrame(
        mat, 
        index=labels,
        columns=labels
    )

    plt.figure(figsize = (n_class,n_class))
    sn.heatmap(df_cm, annot=True,  fmt='g')
    plt.show()


def cross_validate_fullscores(model, dataset, labels, n_class=None, folds=10, txt_labels=None, random_state=42, verbose=True):
    kf = StratifiedKFold(random_state=random_state, n_splits=folds, shuffle=True)  
    total = 0
    totalMat = np.zeros((n_class,n_class))
    n_class = len(np.unique(labels)) if n_class is None else n_class

    metrics = {
        'f1' : [],
        'accuracy': [],
        'precision': [],
        'roc_auc': []
    }

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
        
        mat = confusion_matrix(y_test, result, labels=txt_labels)
        metrics['f1'].append(f1_score(y_test, result, average='weighted'))
        metrics['accuracy'].append(accuracy_score(y_test, result))
        metrics['precision'].append(precision_score(y_test, result, average='weighted'))
        metrics['roc_auc'].append(multiclass_roc_auc_score(y_test, result, average='weighted'))

        totalMat = totalMat + mat
        total = total + sum(y_test==result)

    aggregated_metrics = {
        'f1':        (mean(metrics['f1']), stdev(metrics['f1'])),
        'accuracy':  (mean(metrics['accuracy']), stdev(metrics['accuracy'])),
        'precision': (mean(metrics['precision']), stdev(metrics['precision'])),
        'roc_auc':   (mean(metrics['roc_auc']), stdev(metrics['roc_auc']))
    }

    if verbose:
        print("---\n")
        print("F1 Scores: %0.4f [ +/- %0.4f]" % (aggregated_metrics['f1'][0], aggregated_metrics['f1'][1]))
        print("Accuracy Scores: %0.4f [ +/- %0.4f]" % (aggregated_metrics['accuracy'][0], aggregated_metrics['accuracy'][1]))
        print("Precision Scores: %0.4f [ +/- %0.4f]" % (aggregated_metrics['precision'][0], aggregated_metrics['precision'][1]))
        print("ROC AUC Scores: %0.4f [ +/- %0.4f]" % (aggregated_metrics['roc_auc'][0], aggregated_metrics['roc_auc'][1]))
        print("---\n")
        print(totalMat)

    return [totalMat, total/len(labels), aggregated_metrics]

    
def cross_validate(pl, ds, labels, n_class, show_mat=False, txt_labels=None, random_state=42):
    score = cross_validate_fullscores(pl, ds, labels, n_class=n_class, txt_labels=txt_labels, random_state=random_state)
    if show_mat:
        plot_confusion_matrix(score[0], txt_labels)


def init_simple_classifiers(clf='nc', random_state=42):
    params = {
        'nc': {
            'C':0.484,
            'lr_solver': 'lbfgs',
            'lr_multi_class':'auto',
        },
        'lc' : {
            'C': 0.5,
            'lr_solver': 'lbfgs',
            'lr_multi_class':'auto',
        }
    }

    return [
        ('dt', classifier['tree']()), # dt is so bad, probably should not even be considered
        ('mnb', classifier['mnb']()),
        ('svc', classifier['svc'](C=params[clf]['C'], random_state=random_state)),
        ('lr', classifier['log_reg'](solver=params[clf]['lr_solver'], multi_class=params[clf]['lr_multi_class'], random_state=random_state)),
        # ('knn', KNeighborsClassifier(n_neighbors=10, metric='cosine', weights='distance', n_jobs=-1))
    ]


def init_ensmeta_classifiers(simple_classifier_list, clf='nc', random_state=42):
    params = {
        'nc': {
            'C':0.51,
            'lr_solver': 'lbfgs',
            'lr_multi_class':'auto',
            'bc_estimators': 100,
            'voting': 'hard',
            'ada_estimators': 100,
            'rf_estimators': 100,

            'xgb_estimators': 100,
            'xgb_max_depth': 3,
            'xgb_learning_rate': 0.01,
        },
        'lc' : {
            'C': 1,
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
        ("AdaBoost", classifier['ada'](base_estimator=MultinomialNB(), n_estimators=params[clf]['ada_estimators'])),
        ("RandomForest", classifier['random_forest'](random_state=random_state, n_estimators=params[clf]['rf_estimators'])),
        # ("XGBClassifier", XGBClassifier(max_depth=params[clf]['xgb_max_depth'], n_estimators=params[clf]['xgb_estimators'], learning_rate=params[clf]['xgb_learning_rate'])),
        ("VotingClassifier", VotingClassifier(estimators=simple_classifier_list, voting=params[clf]['voting'])),
        ("BaggingClassifier", BaggingClassifier(base_estimator=classifier['svc'](C=params[clf]['C'], random_state=random_state), 
                                                 n_estimators=params[clf]['bc_estimators'], 
                                                 random_state=random_state)),
    ]


def stacking_classifier(classifiers, random_state=42):
    sclf = StackingCVClassifier(classifiers=[c[1] for c in classifiers], 
                                meta_classifier=LogisticRegression(solver='lbfgs', multi_class='auto', random_state=random_state),
                                use_features_in_secondary=True)

    return Pipeline([
        ('vect', init_vectorizer()),
        ('denser', DenseTransformer()), # StackingCV is not working with Sparse matrix (maybe this is why it sucks so much)
        ('sclf', sclf)
    ])

def test_stacking_classifier(classifiers, ds, labels, plot, n_class, txt_labels, show_mat, random_state=42):
    print("Testing StackingClassifier: \n")
    # building a list of Pipeline vect-classifier
    model = stacking_classifier(classifiers)
    encoded_label = LabelEncoder().fit_transform(labels) # don't know why it doesn't work with string values

    # trying to cross_validate the stack...
    cross_validate(model, ds, encoded_label, n_class, show_mat=show_mat)

    if plot:
        plot_learning_curve(model, "StackingClassifier", ds, encoded_label, random_state=random_state)



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


def build_lc_model(model, feature_selection=None):
    return Pipeline(
        [('union', FeatureUnion([
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
            ])
        )] + 
        ([feature_selection] if feature_selection is not None else []) +
        [model]
    )


def build_nc_model(model, feature_selection=None):
    # feature selection should be a Pair <'selection', SKLEARN_MODEL>
    return Pipeline(
            [('vect', init_vectorizer())] +
            ([('sel', feature_selection)] if feature_selection is not None else []) +
            [('clf', model[1])]
    )


def meta_classify_nc(dataset, show_mat=False, tuning=False, plot=False, load_pretokenized=False):
    # preparing the trainingset
    dataset = shuffle(dataset, random_state=42)

    # avoid wasting time during test...
    # if load_pretokenized:
    #     with open('home/pretokenized_dataset/ncds.pkl', 'rb') as f:
    #         ds = pickle.load(f)
    # else:
    ds = pp.tokenize_list(dataset) # pp.vectorize_list(dataset)  (doc_to_vector stuff, not really working)
        

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
            'clf__C': np.arange(0.001, 1, 0.001) # best C: 0.51
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
        model = build_nc_model(c)

        if tuning:
            print("\nTuning {} Hyper-Parameters with GridSearchCV: \n".format(c[0]))
            tuned_model = GridSearchCV(model, params[c[0]], iid=True,
                scoring='accuracy', cv=StratifiedKFold(random_state=42, shuffle=True, n_splits=10),
                verbose=1,
                n_jobs=-1
            )
            tuned_model.fit(ds, labels)
            print(tuned_model.best_score_, tuned_model.best_params_)
        else:
            print('\n CrossValidation with 10 folds for ' + c[0] + '\n')
            cross_validate(model, ds, labels, n_class, show_mat=show_mat, txt_labels=news_categories, random_state=42)

            if plot:
                plot_learning_curve(model, c[0], ds, labels, random_state=42)

        print('\n---------------------------\n')


    print("\n\nSimple Classifiers WITH Independent Features Selection (chi2-40Percentile):\n")
    for c in classifiers:

        print('\n---------------------------\n')
        # building the model
        model = build_nc_model(c, feature_selection=SelectPercentile(chi2, percentile=40))

        if tuning:
            print("\nTuning {} Hyper-Parameters with GridSearchCV: \n".format(c[0]))
            tuned_model = GridSearchCV(model, params[c[0]], iid=True,
                scoring='accuracy', cv=StratifiedKFold(random_state=42, shuffle=True, n_splits=10),
                verbose=1,
                n_jobs=-1
            )
            tuned_model.fit(ds, labels)
            print(tuned_model.best_score_, tuned_model.best_params_)
        else:
            print('\n CrossValidation with 10 folds for ' + c[0] + '\n')
            cross_validate(model, ds, labels, n_class, show_mat=show_mat, txt_labels=news_categories, random_state=42)

            if plot:
                plot_learning_curve(model, c[0], ds, labels, random_state=42)

        print('\n---------------------------\n')


    print("\n\nEnsembles and Meta-Classifiers:\n")
    
    # set the seed for some classifiers...
    np.random.seed(42)
    
    # init the list...
    ens_meta_classifiers = init_ensmeta_classifiers(classifiers, 'nc')

    for c in ens_meta_classifiers:
        print("\nCV 10 folds - " + c[0])
        # building the pipeline vectorizer-classifier
        model = build_nc_model(c)

        # # Cross_validating the model
        cross_validate(model, ds, labels, n_class, show_mat=show_mat, txt_labels=news_categories, random_state=42)
        if plot:
             plot_learning_curve(model, c[0], ds, labels, random_state=42)


def meta_classify_lc(dataset, show_mat=False, tuning=False, plot=False, load_pretokenized=False):
    # preparing the inputs
    ds = pd.DataFrame()
    dataset = shuffle(dataset, random_state=42)

    # avoid wasting time during test...
    # if load_pretokenized:
    #     with open('home/pretokenized_dataset/lcds.pkl', 'rb') as f:
    #         ds['content'] = pickle.load(f)
    # else:
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

        print('\nCrossValidation with 10 folds for ' + c[0] + '\n')
        cross_validate(pl, ds, labels, 2, show_mat=show_mat, txt_labels=['LIKE', 'DISLIKE'], random_state=42)
        if plot:
            plot_learning_curve(pl, c[0], ds, labels, random_state=42)
        print('\n---------------------------\n')

    print("\n\nEnsembles and Meta-Classifiers:\n")
    np.random.seed(42)

    ens_meta_classifiers = init_ensmeta_classifiers(classifiers, 'lc')

    for c in ens_meta_classifiers:
        print("CV 10 folds - " + c[0])

        # building the pipeline vectorizer-classifier
        pl = build_lc_model(c)
        
        # Cross_validating the model (dunno y its not working with the )
        cross_validate(pl, ds, labels, 2, show_mat=show_mat, txt_labels=['LIKE', 'DISLIKE'], random_state=42)
        if plot:
            plot_learning_curve(pl, c[0], ds, labels, random_state=42)


def t_test(classifiers, X, y, random_state=42, n_repeats=5, n_iter=10, model='nc', alfa=0.05):
    build_model = build_lc_model if model is 'lc' else build_nc_model # this is sooo bad
    pairs = list(itertools.combinations(classifiers, 2))
    results = {}

    for (clf1, clf2) in pairs:
        pair_key = clf1[0]+ '_' + clf2[0]
        print("\n\nTesting " + pair_key)

        for i in range(n_repeats):            
            print(" - Iteration: %d " % (i))
            print(" - random_seed = %d" % (i+random_state))

            results[pair_key] = {}
            results[pair_key]['t_values'] = []
            results[pair_key]['p_values'] = []

            # t-test for the current fold
            t, p = custom_paired_ttest_cv(
                estimator1=build_model(clf1),
                estimator2=build_model(clf2),
                X=X, y=y,
                cv=n_iter,
                random_seed=i+random_state)

            print("    t, p = (%f, %f) (test based on accuracy score of the CV)" % (t, p))            
            results[pair_key]['t_values'].append(t)
            results[pair_key]['p_values'].append(p)
        
        results[pair_key]['fisher'] = stats.combine_pvalues(np.array(results[pair_key]['p_values']))        
        print(" * Combined p_values (Fisher's Method): t, p = (%f, %f)" % (results[pair_key]['fisher'][0], results[pair_key]['fisher'][1]))
        
        if results[pair_key]['fisher'][1] >= alfa:
            print(" ****** T-TEST HAS FAILED! The classifier is better ONLY by chance!")
        else:
            print(" ** T-Test passed!")
            
    return results
    



########## CROSS-VALIDATION + META-CLASSIFICATION #############

    
#####################   MODEL DEPLOY   ##########################

def deploy_news_classifier(dataset, dir_path='home/miner/model'):
    # this function should provide a trained pipeline  (so apparently bagging is the best now)
    clf = BaggingClassifier(base_estimator=classifier['svc'](C=0.51, random_state=42), 
                            n_estimators=100, 
                            random_state=42)

    model = build_nc_model(('clf', clf))

    dataset = shuffle(dataset, random_state=42)
    ds = pp.tokenize_list(dataset)
    labels = dataset['tag'].to_numpy()

    cross_validate_fullscores(model, ds, labels, n_class=9, txt_labels=news_categories)        
    joblib.dump(model, dir_path + '/nws_clf.pkl')
    return model


def deploy_likability_predictor(dataset, dir_path='home/miner/model'):
    # preparing the inputs
    ds = pd.DataFrame()
    dataset = shuffle(dataset, random_state=42)
    ds['content'] = pp.tokenize_list(dataset)
    ds['tag'] = dataset['tag']

    # preparing the targets
    labels = np.asarray([labelize_likability(a)[0] for _,a in dataset.iterrows()])

    # building the model...
    clf = classifier['log_reg'](solver='lbfgs', multi_class='auto', random_state=42)
    model = build_lc_model(('clf', clf))

    cross_validate_fullscores(model, ds, labels, n_class=2, txt_labels=['LIKE', 'DISIKE'])        
    joblib.dump(model, dir_path + '/lik_prd.pkl') 
    return model    


def load_likability_predictor(path='home/miner/model/lik_prd.pkl'):
    model = joblib.load(path)  
    return model


def load_news_classifier(path='home/miner/model/nws_clf.pkl'):
    model = joblib.load(path) 
    return model

#####################   MODEL BUILDING   ##########################


def weka_ttest(classifiers, X, y, repeats=5, n_class=9, txt_labels=None, model='nc'):    
    build_model = build_lc_model if model is 'lc' else build_nc_model # this is sooo bad

    with open("all_metrics_test_" + model + ".arff", "w") as f:
        f.write("@RELATION ExperimentResults\n\n")
        f.write("@ATTRIBUTE Key_Dataset {wekaTest}\n")
        f.write("@ATTRIBUTE Key_Run {1,2,3,4,5}\n")
        f.write("@ATTRIBUTE Key_Scheme {" + ','.join([c[0] for c in classifiers]) + "}\n")
        f.write("@ATTRIBUTE Avg_accuracy numeric\n")
        f.write("@ATTRIBUTE Avg_precision numeric\n")
        f.write("@ATTRIBUTE F1_Score numeric\n")
        f.write("@ATTRIBUTE ROC_Area numeric\n\n")
        f.write("@DATA\n")

        for c in classifiers:
            output = {}
            output['classifier'] = c[0]   
            output['metrics'] = []

            print("Testing %s" % (c[0]))
            for i in range(repeats):
                print(" - Test %d" % (i))
                output['metrics'] = cross_validate_fullscores(build_model(c), X, y, random_state=i, n_class=n_class, verbose=False, txt_labels=txt_labels)[2]

                f.write("wekaTest,%d,%s,%f,%f,%f,%f\n" % (i+1, 
                                                        output['classifier'],
                                                        output['metrics']['accuracy'][0],
                                                        output['metrics']['precision'][0],
                                                        output['metrics']['f1'][0],
                                                        output['metrics']['roc_auc'][0]))
        f.close()
