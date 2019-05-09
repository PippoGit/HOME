import json
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter

import pandas as pd

from db.connector import DBConnector
from miner import classification

def load_config():
    with open("config/config.json") as f:
        config = json.load(f)
    return config

flatten = lambda l: [item for sublist in l for item in sublist]

def show_wordcloud(dataset):
    word_cloud_dict = Counter(dataset)
    wordcloud = WordCloud(width = 1000, height = 500).generate_from_frequencies(word_cloud_dict)
    plt.figure(figsize=(15,8))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()

def deploy_models():
    # importing configuration 
    print("\nimporting config file...") 
    config = load_config()

    # preparing the components
    db = DBConnector(**config['db'])

    print("building the news classifier...")
    classification.deploy_news_classifier(pd.DataFrame(db.find_trainingset()))
    print("building the likability predictor...")
    classification.deploy_likability_predictor(pd.DataFrame(db.find_likabilityset()))
    print("models built!")