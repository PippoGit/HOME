import pandas as pd
from home import utility
from home.miner import classification
from home.db.connector import DBConnector
from home.miner import preprocessing
from sklearn.utils import shuffle

import nltk


# plot
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sn

import pickle


def classify(nc=True, lc=True, show_mat=False, tuning=False, plot_learning_curve=False, pretokenized=False):
     # importing configuration 
    print("\nimporting config file...") 
    config = utility.load_config()

    # preparing the components
    print("\npreparing the components...\n")
    db = DBConnector(**config['db'])

    if nc:
        print('\nmeta-classifing NC ... ')
        classification.meta_classify_nc(dataset=pd.DataFrame(db.find_trainingset()), show_mat=show_mat, tuning=tuning, plot=plot_learning_curve, load_pretokenized=pretokenized)
    
    if lc: 
        print('\nmeta-classifing LC ... ')
        classification.meta_classify_lc(dataset=pd.DataFrame(db.find_likabilityset()), tuning=tuning, show_mat=show_mat, plot=plot_learning_curve, load_pretokenized=pretokenized)


def deploy_models(path='home/miner/model'):
    # importing configuration 
    print("\nimporting config file...") 
    config = utility.load_config()

    # preparing the components
    print("\npreparing the components...\n")
    db = DBConnector(**config['db'])

    print("building the news classifier...")
    classification.deploy_news_classifier(pd.DataFrame(db.find_trainingset()), path)
    print("building the likability predictor...")
    classification.deploy_likability_predictor(pd.DataFrame(db.find_likabilityset()), path)
    print("models built!")


def t_test_nc(model='nc', load_pretokenized=False):
    # importing configuration 
    print("\nimporting config file...") 
    config = utility.load_config()

    # preparing the components
    print("\npreparing the components...\n")
    db = DBConnector(**config['db'])
    dataset = shuffle(pd.DataFrame(db.find_trainingset()), random_state=42)

    print("\ntokenizing...\n")
    ds = preprocessing.tokenize_list(dataset) # pp.vectorize_list(dataset)  (doc_to_vector stuff, not really working)
    # ds = pd.DataFrame(ds) # why do i need this? t-test function from mlxtend is strange...q

    # preparing the targets
    labels = dataset['tag'].to_numpy()

    # t test here!
    print("\n\nt-testing %s ...\n\n" % (model))
    classifiers = classification.init_simple_classifiers(model) + classification.init_ensmeta_classifiers(model)
    results = classification.t_test(classifiers, ds, labels, model=model)
    
    # dumping the results (...)
    with open('t_test_scores.pkl', 'wb') as f:
        pickle.dump(results, f)