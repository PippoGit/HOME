import spacy
import nltk
import pandas as pd
from unidecode import unidecode

nlp = spacy.load('it_core_news_sm')
custom_sw = set(line.strip() for line in open('config/stopwords-it.txt'))
stopwords = set(nltk.corpus.stopwords.words('italian')).union(custom_sw)

stemmer = {
    'porter': nltk.stem.PorterStemmer,
    'snowball': nltk.stem.snowball.ItalianStemmer
}

def word_tokenize(corpus, stm='snowball'):
    # preparing stuff
    st = stemmer[stm]()
    tokens = nlp(corpus) # this is really slow, but can't avoid using it!

    # tokenization + stemming
    tokens = [st.stem(unidecode(t.norm_)) for t in tokens if not (t.is_punct or t.is_space or t.like_num)
                                                             and unidecode(t.norm_) not in stopwords
                                                             and len(t.text) >= 2]
    return tokens


def tokenize_article(article, should_remove_duplicates=False):
    corpus = article['title'] + '\n' + article['description']
    tokens = word_tokenize(corpus)
    return list(set(tokens)) if should_remove_duplicates else tokens


def tokenize_likability(article):
    corpus = article['title']
    return word_tokenize(corpus)


def tokenize_list(articles, should_remove_duplicates=False):
    if type(articles) is list:
        articles = pd.DataFrame(articles)
    return [tokenize_article(a, should_remove_duplicates=should_remove_duplicates) for _,a in articles.iterrows()]