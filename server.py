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
# print(db.find_liked())

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
    def get(self, num_articles=50):
        return newsfeed.training_samples(num_articles), 200


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


class LikedArticles(Resource):
    def get(self):
        return db.find_liked(), 200


class DislikedArticles(Resource):
    def get(self):
        return db.find_disliked(), 200


class ReadArticles(Resource):
    def get(self):
        return db.find_read(), 200


# rest API routes
api.add_resource(Feed, "/api/feed", "/api/feed/<int:num_articles>")
api.add_resource(Learn, "/api/learn", "/api/learn/<int:num_articles>")

api.add_resource(Like, "/api/like")
api.add_resource(Dislike, "/api/dislike")
api.add_resource(Read, "/api/read")

api.add_resource(LikedArticles, "/api/liked_articles")
api.add_resource(DislikedArticles, "/api/disliked_articles")
api.add_resource(ReadArticles, "/api/read_articles")


# front-end routes
@app.route('/')
def render_home():
    return render_template('index.html')


# main
if __name__ == "__main__":
    print("\nstarting server...")
    app.run(debug=True)