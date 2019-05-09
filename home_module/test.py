import pandas as pd
import utility
from miner import classification
from db.connector import DBConnector


def test_classifiers(nc=True, lc=True, show_mat=False, tuning=False):
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
