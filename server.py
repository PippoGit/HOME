import home

from flask import Flask, render_template, request
from flask_restful import Api, Resource, reqparse
from flask_cors import CORS

# importing configuration 
print("\nimporting config file...") 
config = home.load_config()

# preparing the components
db = home.DBConnector(**config['db'])
newsfeed = home.NewsFeed(config['feeds'])
miner = home.Miner()

# loading initial feed
print("\nloading feeds...") 
newsfeed.load()

# filtering the dataset using some machinelearning magic...


# inserting the results into the db


# building Flask
app = Flask(__name__,
            static_url_path='', 
            static_folder='public/static',
            template_folder='public/template')
api = Api(app)
CORS(app)

# REST Resources
class Feed(Resource):
    def get(self, num_articles=None):
        # getting the best articles from the db...
        return 400 # Not available yet!
    
    def put(self):
        newsfeed.load()
        return 200


class Learn(Resource):
    def get(self):
        return newsfeed.training_samples(50), 200


class Like(Resource):
    def post(self):
        article = request.get_json()
        db.change_like_dislike(article, like=True)
        return 200
    
    def delete(self):
        article = request.get_json()
        db.change_like_dislike(article)
        return 200


class Dislike(Resource):
    def post(self):
        article = request.get_json()
        db.change_like_dislike(article, dislike=True)
        return 200
    
    def delete(self):
        article = request.get_json()
        db.change_like_dislike(article)
        return 200


class Read(Resource):
    def post(self):
        article = request.get_json()
        db.insert_read(article)
        return 200


# rest API routes
api.add_resource(Feed, "/api/feed", "/api/feed/<int:num_articles>")
api.add_resource(Learn, "/api/learn")
api.add_resource(Like, "/api/like")
api.add_resource(Dislike, "/api/dislike")
api.add_resource(Read, "/api/read")


# front-end routes
@app.route('/')
def render_home():
    return render_template('index.html')


# main
if __name__ == "__main__":
    print("\nstarting server...")
    app.run(debug=True)