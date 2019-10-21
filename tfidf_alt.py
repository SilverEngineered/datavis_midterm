from bow import *
from tfidf import *
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

odd_ends = ['<', '>', '', '!', '@', '#', ',', '.', "|", "max>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'",
            '|>']

# creating bag of words
bow = get_bow_list(True)
tf = Counter(bow)
print("# of Unique Words:%s" % len(tf))
sorted_tf = sorted(tf.items(), key=lambda x: x[1])
sorted_tf = list(filter(lambda x: x[0] not in odd_ends, sorted_tf))

# creating tfidf
newsgroups = fetch_20newsgroups(remove=('headers', 'footers'),categories=None)
corpus = list(newsgroups.data)
sorted_tfidf = []
corpusDocSize = len(corpus)
for count, pair in enumerate(tqdm.tqdm(sorted_tf[-2000:])):
    appearances = 0
    for doc in corpus:
        if pair[0] in doc.lower():
            appearances += 1
            continue
    idf = np.log(corpusDocSize/(appearances + 1))
    sorted_tfidf.append((pair[0], pair[1]/len(tf) * idf))
sorted_tfidf = sorted(dict(sorted_tfidf).items(), key=lambda x: x[1])
sorted_tfidf = list(filter(lambda x: x[0] not in odd_ends, sorted_tfidf))

# tf graph
for pair in reversed(sorted_tf[-20:]):
    plt.bar(pair[0], pair[1]/len(tf))
plt.suptitle('Top Words in Corpus')
plt.ylabel("Term Frequency")
plt.show()

# tfidf graph
for pair in reversed(sorted_tfidf[-20:]):
    plt.bar(pair[0], pair[1])
plt.suptitle('Top Words in Corpus')
plt.ylabel("TF-IDF")
plt.show()
