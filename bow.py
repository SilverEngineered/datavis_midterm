from gensim import corpora
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

vectorizer = TfidfVectorizer()
newsgroups = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')
texts = list(newsgroups.data)
targets = newsgroups.target

vectors = vectorizer.fit_transform(texts) #TFIDF
vectors_test = vectorizer.transform(newsgroups_test.data)

#Naive Bayes classifier
clf = MultinomialNB(alpha=.01)
clf.fit(vectors, targets)

pred = clf.predict(vectors_test)
#Get accuracy
acc = metrics.f1_score(newsgroups_test.target, pred, average='macro')
print(acc)