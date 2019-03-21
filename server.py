import feedparser
import json
from flask import Flask, render_template
from flask_restful import Api, Resource, reqparse
from flask_cors import CORS
from bs4 import BeautifulSoup

app = Flask(__name__,
    static_url_path='', 
    static_folder='public/static',
    template_folder='public/template')
api = Api(app)
CORS(app)

# importing dataset
print("\nimporting config file...") 
with open("config/config.json") as f:
    config = json.load(f)

print("\nloading feeds...") 
news = []
for source in config["feeds"]:
    entries = feedparser.parse(source['url']).entries

    for e in entries:
        #value = e.summary[0]['value']
        soup = BeautifulSoup(e['summary'], features="html.parser")
        imgurl = soup.find('img')

        news.append({
            'title' : e['title'] if ('title' in e) else "",
            'author': e['author'] if ('author' in e) else "",
            'description' : soup.text if soup is not None else "",
            'datetime' : e['published'] if ('published' in e) else "",
            'img' : imgurl['src'] if imgurl is not None else "",
            'link': e['link'] if ('link' in e) else "",
            'source' : source['name']
        })

    # news.extend(entries)
    # news = sorted(news, key=lambda entry: entry["published"])
    # news.reverse()


# REST Resources
class News(Resource):
    def get(self, num_pages=None):
        if num_pages is not None:
            return news[:num_pages], 200
        else:
            return news, 200

    def post(self, name):
        parser = reqparse.RequestParser()
        parser.add_argument("name")
        parser.add_argument("occupation")
        args = parser.parse_args()

        return args, 201

# rest routes
api.add_resource(News, "/api/news", "/api/news/<int:num_pages>")

# frontend routes
@app.route('/')
def render_home():
    return render_template('index.html', name="Nyca")


# main
if __name__ == "__main__":
    print("\nstarting server...")
    app.run(debug=True)