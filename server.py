import home

from flask import Flask, render_template, request
from flask_restful import Api, Resource, reqparse
from flask_cors import CORS

# importing configuration 
print("\nimporting config file...") 
config = home.load_config()

# preparing the components
db = home.DBConnector(**config['db'])
feed_parser = home.Parser(config['feeds']) # should the feed be a Pandas Dataframe too? dunno
newsfeed = home.NewsFeed() # i don't know if i actually need this class (maybe a list will be fine)
                           # even better: i could use Pandas Dataframe => a lot easier to use!

# loading initial feed
print("\nloading feeds...") 
feed_parser.parse()

# filtering the dataset using some machinelearning magic...
miner = home.Miner(feed_parser.parsed_feed)

# inserting the results into the db
# print(db.find_liked())

# building Flask
app = Flask(__name__,
            static_url_path='', 
            static_folder='public/static',
            template_folder='public/template')
api = Api(app)
CORS(app, origins=['http://localhost:5000', 'http://imac.local:5000', 'http://127.0.0.1:5000', "http://192.168.1.200:5000"])

# REST Resources
class Feed(Resource):
    def get(self, num_articles=None):
        # getting the best articles from the db...

        # miner.update_dataset(feed_parser.parsed_feed)
        # miner.fix_null()
        tokens = miner.tokenize(filter=True)
        tokens_nsw = miner.remove_stopwords(tokens)

        return tokens_nsw # Not available yet!
    
    def patch(self):
        # update the sources => Parse again RSS
        feed_parser.parse()
        # feed_parser.sort_feed()

        # re-train the model

        # re-apply the filter

        # return filtered feed (NOT the feed_parser stuff)

        return 200


class Learn(Resource):
    def get(self, num_articles=50):
        return feed_parser.training_samples(num_articles), 200


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


class Tag(Resource):
    def get(self):
        return db.find_untagged(), 200
    
    def put(self):
        article = request.get_json()
        db.tag_article(article['_id'], article['tag'])
        return 200


# rest API routes
api.add_resource(Feed, "/api/feed", "/api/feed/<int:num_articles>")
api.add_resource(Learn, "/api/learn", "/api/learn/<int:num_articles>")

api.add_resource(Like, "/api/like")
api.add_resource(Dislike, "/api/dislike")
api.add_resource(Read, "/api/read")

api.add_resource(LikedArticles, "/api/liked_articles")
api.add_resource(DislikedArticles, "/api/disliked_articles")
api.add_resource(ReadArticles, "/api/read_articles")

api.add_resource(Tag, "/api/tag")


# front-end routes
@app.route('/')
def render_home():
    return render_template('index.html')


# main
if __name__ == "__main__":
    print("\nstarting server...")
    app.run(debug=True)