import home

from flask import Flask, render_template, request
from flask_restful import Api, Resource, reqparse
from flask_cors import CORS

# importing configuration 
print("\nimporting config file...") 
config = home.load_config()

# preparing the components
db = home.DBConnector(**config['db'])
feed_parser = home.Parser(config['feeds'])
                          
# loading models
print("\nloading the models...") 
news_classifier = home.Miner.load_news_classifier()
likability_predictor = home.Miner.load_likability_predictor()

# building the newsfeed...
print("\nparsing the news...") 
newsfeed = home.NewsFeed(news_classifier, likability_predictor)
feed_parser.parse()

print("\nbuilding the feed...") 
newsfeed.build_feed(feed_parser.parsed_feed)

print("Custom newsfeed built!")

# building Flask
app = Flask(__name__,
            static_url_path='', 
            static_folder='public/static',
            template_folder='public/template')
api = Api(app)
CORS(app, origins=['http://localhost:5000', 'http://imac.local:5000', 'http://127.0.0.1:5000', "http://192.168.1.200:5000"])

# REST Resources
class Feed(Resource):
    def get(self, descriptor=None):
        if descriptor=='learn':
            return  feed_parser.sorted_feed(), 200 # feed_parser.training_samples(168), 200
        return (newsfeed.to_list(), 200) if descriptor is None else (db.find_feed(descriptor), 200)


    def patch(self):
        # update the sources => Parse again RSS
        feed_parser.parse() 

        # update the newsfeed
        newsfeed.build_feed(feed_parser.parsed_feed)

        # send back the data
        return (newsfeed.to_list(), 200)


class Like(Resource):
    def post(self):
        article = request.get_json()
        db.update_article(article, {'dislike':False, 'like':True})
        return 200
    
    def delete(self):
        article = request.get_json()
        db.update_article(article, {'dislike':False, 'like':False})
        return 200


class Dislike(Resource):
    def post(self):
        article = request.get_json()
        db.update_article(article, {'dislike':True, 'like':False} )
        return 200
    
    def delete(self):
        article = request.get_json()
        db.update_article(article, {'dislike':False, 'like':False})
        return 200


class Read(Resource):
    def post(self):
        article = request.get_json()
        db.update_article(article, {'read':True})
        return 200


class Tag(Resource):
    def get(self):
        return db.find_untagged(), 200
    
    def put(self):
        article = request.get_json()
        db.tag_article(article['_id'], article['tag'])
        return 200


class Model(Resource):
    def patch(self):
        # fit again the model (TBD)  
        pass
        

class Statistics(Resource):
    def get(self, distribution=None):
        return db.stats(distribution), 200


# rest API routes
api.add_resource(Feed, "/api/feed", "/api/feed/<string:descriptor>")

api.add_resource(Like, "/api/like")
api.add_resource(Dislike, "/api/dislike")
api.add_resource(Read, "/api/read")
api.add_resource(Tag, "/api/tag")

api.add_resource(Model, "/api/model")

api.add_resource(Statistics, "/api/stats/<string:distribution>")


# front-end routes
@app.route('/')
def render_home():
    return render_template('index.html')


# main
if __name__ == "__main__":
    print("\nstarting server...")
    app.run(debug=True)