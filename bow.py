from gensim import corpora
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
# from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from nltk.stem import WordNetLemmatizer
import re
import nltk
# nltk.download('all')
lemmatizer = WordNetLemmatizer()

def get_bow(cleaned=False):
    vectorizer = CountVectorizer()
    if cleaned:
        newsgroups = fetch_20newsgroups(remove=('headers', 'footers'))
    else:
        newsgroups = fetch_20newsgroups()
    texts = list(newsgroups.data)
    bow = vectorizer.fit_transform(texts)
    return bow

def get_bow_list(cleaned=False,categories=None):
    if cleaned:
        newsgroups = fetch_20newsgroups(remove=('headers', 'footers'),categories=categories)
    else:
        newsgroups = fetch_20newsgroups()
    corpus = list(newsgroups.data)
    print("# of Documents:%s" % len(corpus))
    sentences = []
    for document in corpus:
        sentences.append(re.split(" |\n", document.lower()))
    bow = []
    for word in [val for sublist in sentences for val in sublist]:
        bow.append(lemmatizer.lemmatize(word))
    print("# of Total Words:%s" % len(bow))
    return bow