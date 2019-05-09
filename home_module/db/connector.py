import pymongo
import datetime
import random

def datetime_to_string(article):
    article['datetime'] = str(article['datetime'])
    return article


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
                'tag__':1, 
                'like':1, 
                'dislike':1, 
                'read':1
            }
        ).sort([('tag',1)]))
        return results


    def find_likabilityset(self):
        articles = self.db['articles']
        results = list(articles.find({
                'tag': {'$exists':True},
                '$or': [{ 'like': True }, { 'dislike': True }]
            }, {
                'title':1, 
                'description':1, 
                'tag':1, 
                'source': 1,
                'tag__':1,
                'like':1, 
                'dislike':1, 
                'read':1
            }
        ).sort([('dislike',1)]))
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
            'num_likes': self.db.articles.count({'like':True}),
            'num_dislikes': self.db.articles.count({'dislike':True}),
            'num_read': self.db.articles.count({'read':True}),
            'num_ignored': self.db.articles.count({'like':False, 'dislike':False, 'read':False}),
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
            'training': self.find_trainingset,
            'training_likability': self.find_likabilityset
        }
        return feed_descriptors[descriptor]()


    def close(self):
        self.client.close()
