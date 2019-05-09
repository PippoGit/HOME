import hashlib
import datetime
import ssl
import random

import feedparser
from bs4 import BeautifulSoup


def try_parse_date(article_date):
    now = datetime.datetime.now()
    dt = datetime.datetime.strptime(article_date, "%Y-%m-%dT%H:%M:%S")  if article_date is not None else datetime.datetime(now.year, now.month, now.day)
    return dt


# RSS Parser class
class Parser:
    def __init__(self, sources):
        self.sources = sources
        self.parsed_feed = []


    def parse(self):
        self.parsed_feed = []
        curr_count = 0

        if hasattr(ssl, '_create_unverified_context'):
            ssl._create_default_https_context = ssl._create_unverified_context

        for source in self.sources:
            parsed = feedparser.parse(source['url'])
            if parsed.bozo == 1:
                print(parsed.bozo_exception)

            entries = parsed.entries
            
            for e in entries:
                # parsing the content of the summary
                if 'summary' not in e: # if the entry has no summary key just skip it... (it won't happen that often)
                    continue
                soup = BeautifulSoup(e['summary'], features="html.parser")
                imgurl = soup.find('img')

                article_date = datetime.datetime(*e['published_parsed'][:6]) if ('published_parsed' in e) else None
                if article_date is not None and ((datetime.datetime.now() - article_date) > datetime.timedelta(hours=12)):
                    continue  # skip old articles...

                # building the article 
                article = {
                    'title' : e['title'] if ('title' in e) else "",
                    'author': e['author'] if ('author' in e) else "",
                    'description' : soup.text if soup is not None else "",
                    'datetime' : article_date.isoformat() if article_date is not None else None,
                    'img' : imgurl['src'] if (imgurl is not None and imgurl.has_attr('src')) else "",
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
                self.parsed_feed.append(article)

            print("{} loaded! ({} entries)".format(source['name'], len(self.parsed_feed) - curr_count))
            curr_count = len(self.parsed_feed)

        print("whole newsfeed loaded ({} entries)".format(len(self.parsed_feed)))


    def training_samples(self, num_articles=50):
        return random.sample(self.parsed_feed, num_articles)


    def sorted_feed(self):
        return sorted(self.parsed_feed, key=lambda x: try_parse_date(x['datetime']), reverse=True)
