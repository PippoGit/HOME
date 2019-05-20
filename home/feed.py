import pandas as pd
from home.miner import preprocessing as pp

# NewsFeed class
class NewsFeed:
    def __init__(self, nws_clf, lik_prd):
        self.nws_clf = nws_clf
        self.lik_prd = lik_prd
        self.feed = None


    def to_list(self):
        # sorted_feed = self.feed.sort_values(by=['datetime'], ascending=False)
        return self.feed.to_dict('records')


    def build_feed(self, parsed_feed):
        self.feed = pd.DataFrame(parsed_feed)

        features = pd.DataFrame()        
        features['content']     = pp.tokenize_list(parsed_feed)
        features['tag']         = self.nws_clf.predict(features['content'])

        self.feed['likability'] = self.lik_prd.predict(features)
        self.feed['predicted_tag'] = features['tag']

        self.feed = self.feed[self.feed['likability'] == 'LIKE'].drop('likability', axis=1)
