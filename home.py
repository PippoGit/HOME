import feedparser
from bs4 import BeautifulSoup
import hashlib, datetime, ssl, random, json
import pymongo

# some util function 
def load_config():
    with open("config/config.json") as f:
        config = json.load(f)
    return config


# NewsFeed class
class NewsFeed:
    def __init__(self, sources):
        self.sources = sources
        self.feed = []


    def load(self):
        if hasattr(ssl, '_create_unverified_context'):
            ssl._create_default_https_context = ssl._create_unverified_context

        for source in self.sources:
            parsed = feedparser.parse(source['url'])
            if parsed.bozo == 1:
                print(parsed.bozo_exception)

            entries = parsed.entries
            
            for e in entries:
                # parsing the content of the summary
                soup = BeautifulSoup(e['summary'], features="html.parser")
                imgurl = soup.find('img')

                # building the article
                article = {
                    'title' : e['title'] if ('title' in e) else "",
                    'author': e['author'] if ('author' in e) else "",
                    'description' : soup.text if soup is not None else "",
                    'datetime' : e['published'][:-6] if ('published' in e) else "",
                    'img' : imgurl['src'] if imgurl is not None else "",
                    'link': e['link'] if ('link' in e) else "",
                    'source' : source['name'],

                    'like' : False,
                    'dislike' : False,
                    'read' : False
                }

                # adding the id
                ida = ''.join('{}{}'.format(key, val) for key, val in article.items())
                article['_id'] = hashlib.sha1(ida.encode()).hexdigest()
                
                # feed the feeder
                self.feed.append(article)

            print("{} loaded! ({} entries)".format(source['name'], len(entries)))

        print("whole newsfeed loaded ({} entries)".format(len(self.feed)))

    
    def sorted_feed(self, num_articles=None):
        feed = self.feed[:num_articles] if num_articles is not None else self.feed
        feed = sorted(feed, key=lambda kv: datetime.datetime.strptime(kv['datetime'], '%a, %d %b %Y %H:%M:%S'), reverse=True)
        return feed


    def training_samples(self, num_articles=50):
        return random.sample(self.feed, num_articles)

# DataMining stuff HERE!
class Miner:
  pass

# MongoDB connector
class DBConnector:
    def __init__(self, host, name, user=None, password=None):
        # connecting with mongodb
        self.client = pymongo.MongoClient(host)
        self.db = self.client[name]

    def update_article(self, article, values):
        articles = self.db['articles']
        
        for k, v in values.items():
            article[k] = v

        results = articles.find_one({'_id':article['_id']})
        if results is None:
            # insert article
            articles.insert(article)
        else:
            # update old article
            articles.update(
                {'_id':article['_id']},
                { '$set': values }
            )

    def tag_article(self, article_id, tag):
        articles = self.db['articles']
        articles.update(
            {'_id':article_id},
            {'$set':{'tag': tag}}
        )


    def find(self, query):
        articles = self.db['articles']
        results = articles.find(query)
        return list(results)

    def find_one(self, query):
        articles = self.db['articles']
        return articles.find_one(query)
        
    def find_liked(self):
        return self.find({'like':True})

    def find_disliked(self):
        return self.find({'dislike':True})

    def find_read(self):
        return self.find({'read':True})

    def find_untagged(self):
        return self.find_one({'tag':None})

    def close(self):
        pass
    
    def insert_feed(self, feed):
        pass


