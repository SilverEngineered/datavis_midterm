from gensim import corpora
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
from sklearn.svm import SVC

def get_bow(cleaned=False):
	vectorizer = CountVectorizer()
	if cleaned:
		newsgroups = fetch_20newsgroups(remove=('headers', 'footers'))
	else:
		newsgroups = fetch_20newsgroups()
	texts = list(newsgroups.data)
	bow = vectorizer.fit_transform(texts)
	return bow
