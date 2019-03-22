import json

def load_config():
    with open("config/config.json") as f:
        config = json.load(f)
    return config


# init with list of urls
class NewsFeed:
  pass

class Miner:
  pass

class DBConnector:
  pass


