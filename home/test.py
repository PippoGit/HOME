import pandas as pd
from home import utility
from home.miner import classification
from home.db.connector import DBConnector
from home.miner import preprocessing
from sklearn.utils import shuffle

import nltk
import numpy as np


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


def weka_arff_ttest(model='nc'):
	# importing config
	print("\nimporting config file...") 
	config = utility.load_config()
	
	print("\npreparing the components...\n")
	db = DBConnector(**config['db'])

	if model is 'nc':
		n_class = 9
		dataset = shuffle(pd.DataFrame(db.find_trainingset()), random_state=42)
		# dataset = dataset.head(100)
		print("\ntokenizing...\n")
		ds = preprocessing.tokenize_list(dataset)
		labels = dataset['tag'].to_numpy()

		txt_labels = classification.news_categories
	else:
		n_class = 2
		dataset = shuffle(pd.DataFrame(db.find_likabilityset()), random_state=42)

		print("\ntokenizing...\n")
		ds = pd.DataFrame()
		ds['content'] = preprocessing.tokenize_list(dataset)
		ds['tag'] = dataset['tag']

		labels = np.asarray([classification.labelize_likability(a)[0] for _,a in dataset.iterrows()])
		txt_labels = ['LIKE', 'DISLIKE']

	# preparing classifiers
	print("\n\nt-testing %s ...\n\n" % (model))
	classifiers = classification.init_simple_classifiers(model)
	classifiers = classifiers + classification.init_ensmeta_classifiers(classifiers, model) # putting together simple and ens/meta

	classification.weka_ttest(classifiers, ds, labels, n_class=n_class, txt_labels=txt_labels, model=model)
