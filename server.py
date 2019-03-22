import home

from flask import Flask, render_template
from flask_restful import Api, Resource, reqparse
from flask_cors import CORS

import feedparser
from bs4 import BeautifulSoup
import hashlib

# importing configuration 
print("\nimporting config file...") 
config = home.load_config()

# preparing the components
db = home.DBConnector()
newsfeed = home.NewsFeed()
miner = home.Miner()

# loading initial feed
print("\nloading feeds...") 
news = []
for source in config["feeds"]:
    entries = feedparser.parse(source['url']).entries

    for e in entries:
        #value = e.summary[0]['value']
        soup = BeautifulSoup(e['summary'], features="html.parser")
        imgurl = soup.find('img')
        idn = e['title'] + source['name']

        news.append({
            'id' : hashlib.sha1(idn.encode()).hexdigest(),
            'title' : e['title'] if ('title' in e) else "",
            'author': e['author'] if ('author' in e) else "",
            'description' : soup.text if soup is not None else "",
            'datetime' : e['published'][:-5] if ('published' in e) else "",
            'img' : imgurl['src'] if imgurl is not None else "",
            'link': e['link'] if ('link' in e) else "",
            'source' : source['name']
        })
    # news = sorted(news, key=lambda entry: entry["published"])
    # news.reverse()

# building Flask
app = Flask(__name__,
            static_url_path='', 
            static_folder='public/static',
            template_folder='public/template')
api = Api(app)
CORS(app)

# REST Resources
class Feed(Resource):
    def get(self, num_pages=None):
        if num_pages is not None:
            return news[:num_pages], 200
        else:
            return news, 200


# rest routes
api.add_resource(Feed, "/api/feed", "/api/feed/<int:num_pages>")

# frontend routes
@app.route('/')
def render_home():
    return render_template('index.html')


# main
if __name__ == "__main__":
    print("\nstarting server...")
    app.run(debug=True)