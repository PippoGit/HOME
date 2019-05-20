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



def classify(nc=True, lc=True, show_mat=False, tuning=False, plot_learning_curve=False):
     # importing configuration 
    print("\nimporting config file...") 
    config = utility.load_config()

    # preparing the components
    print("\npreparing the components...\n")
    db = DBConnector(**config['db'])

    if nc:
        print('\nmeta-classifing NC ... ')
        classification.meta_classify_nc(dataset=pd.DataFrame(db.find_trainingset()), show_mat=show_mat, tuning=tuning, plot=plot_learning_curve)
    
    if lc: 
        print('\nmeta-classifing LC ... ')
        classification.meta_classify_lc(dataset=pd.DataFrame(db.find_likabilityset()), tuning=tuning, show_mat=show_mat, plot=plot_learning_curve)


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


def t_test_nc():
    # importing configuration 
    print("\nimporting config file...") 
    config = utility.load_config()

    # preparing the components
    print("\npreparing the components...\n")
    db = DBConnector(**config['db'])

    dataset= shuffle(pd.DataFrame(db.find_trainingset()), random_state=42)
    ds = preprocessing.tokenize_list(dataset) # pp.vectorize_list(dataset)  (doc_to_vector stuff, not really working)

    # preparing the targets
    labels = dataset['tag'].to_numpy()

    # t test here!
    classifiers = classification.init_simple_classifiers('nc')
    classification.t_test(classifiers, ds, labels)


# TODO: plot word distribution for each category (just to see if it makes sense (indeed it does))
# (i think i'm just going to skip this)
# def plot_word_tag_distribution():
#     # importing configuration 
#     config = utility.load_config()

#     # preparing the components
#     db = DBConnector(**config['db'])
#     dataset = pd.DataFrame(db.find_trainingset())

#     for cat in classification.news_categories:
#         print("info about %s" % (cat))
#         data = dataset[dataset['tag'] == cat]
#         data['tokens'] = [preprocessing.tokenize_article(a) for _,a in data.iterrows()]
#         words = [word for word in [article['tokens'] for _,article in data.iterrows()]]
#         freqdist = nltk.FreqDist(words)
#         freqdist.plot(50)
#         plt.show()

