import pandas as pd
from home import utility
from home.miner import classification
from home.db.connector import DBConnector


def classify(nc=True, lc=True, show_mat=False, tuning=False):
     # importing configuration 
    print("\nimporting config file...") 
    config = utility.load_config()

    # preparing the components
    print("\npreparing the components...\n")
    db = DBConnector(**config['db'])

    if nc:
        print('\nmeta-classifing NC ... ')
        classification.meta_classify_nc(dataset=pd.DataFrame(db.find_trainingset()), show_mat=show_mat, tuning=tuning)
    
    if lc: 
        print('\nmeta-classifing LC ... ')
        classification.meta_classify_lc(dataset=pd.DataFrame(db.find_likabilityset()), tuning=tuning, show_mat=show_mat)


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