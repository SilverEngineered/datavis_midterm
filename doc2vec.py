from bow import *
from tfidf import *
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from sklearn.manifold import TSNE
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

odd_ends = ['<', '>', '', '!', '@', '#', ',', '.', "|", "max>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'",
            '|>']

# bag of words again
bow = get_bow_list(True)
tf = Counter(bow)
print("# of Unique Words:%s" % len(tf))
sorted_tf = sorted(tf.items(), key=lambda x: x[1])
sorted_tf = list(filter(lambda x: x[0] not in odd_ends, sorted_tf))

# collect some unique words
uniqueWords = []
for pair in sorted_tf:
    uniqueWords.append(pair[1])
newsgroups = fetch_20newsgroups(remove=('headers', 'footers'),categories=None)
corpus = list(newsgroups.data)
# begin model training
tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(corpus)]
max_epochs = 20
vec_size = 20
alpha = 0.025
model1 = Doc2Vec(size=vec_size,
                alpha=alpha,
                min_alpha=0.00025,
                min_count=1,
                dm=1)
model1.build_vocab(tagged_data)
for epoch in tqdm.tqdm(range(max_epochs)):
    model1.train(tagged_data,
                total_examples=model1.corpus_count,
                epochs=model1.iter)
    # decrease the learning rate
    model1.alpha -= 0.0002
    # fix the learning rate, no decay
    model1.min_alpha = model1.alpha
model1.save("d2v_1.model")
print("Model Saved")
# train second model
tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(corpus)]
max_epochs = 30
vec_size = 20
alpha = 0.025
model1 = Doc2Vec(size=vec_size,
                alpha=alpha,
                min_alpha=0.00025,
                min_count=1,
                dm=1)
model1.build_vocab(tagged_data)
for epoch in tqdm.tqdm(range(max_epochs)):
    model1.train(tagged_data,
                total_examples=model1.corpus_count,
                epochs=model1.iter)
    # decrease the learning rate
    model1.alpha -= 0.0002
    # fix the learning rate, no decay
    model1.min_alpha = model1.alpha
model1.save("d2v_2.model")
print("Model Saved")
# use first model
model = Doc2Vec.load("d2v_1.model")
embedded_words = []
for pair in reversed(sorted_tf[:]):
    try:
        embedded_words.append((pair[0],model[pair[0]]))
    except:
        continue

for pair in embedded_words:
    plt.scatter(x = pair[1][0], y = pair[1][1], c='b', alpha=.5)
    plt.annotate(pair[0], (pair[1][0], pair[1][1]), fontsize=8)
plt.suptitle('Embedded Word Association')
plt.show()

aquatic = model.similar_by_word("aquatic", topn=200)
cooking = model.similar_by_word("cooking", topn=200)
desert = model.similar_by_word("desert", topn=200)
flight = model.similar_by_word("flight", topn=200)
vegetarian = model.similar_by_word("vegetarian", topn=200)

for pair in embedded_words:
    for word in aquatic:
        if pair[0] == word[0]:
            plt.scatter(x = pair[1][0], y = pair[1][1], c='blue', alpha=.5)
    for word in cooking:
        if pair[0] == word[0]:
            plt.scatter(x = pair[1][0], y = pair[1][1], c='red', alpha=.5)
    for word in desert:
        if pair[0] == word[0]:
            plt.scatter(x = pair[1][0], y = pair[1][1], c='orange', alpha=.5)
    for word in flight:
        if pair[0] == word[0]:
            plt.scatter(x = pair[1][0], y = pair[1][1], c='gray', alpha=.5)
    for word in vegetarian:
        if pair[0] == word[0]:
            plt.scatter(x = pair[1][0], y = pair[1][1], c='green', alpha=.5)
plt.suptitle('Document Embedding')
blue_patch = mpatches.Patch(color='blue', label='The blue data')
red_patch = mpatches.Patch(color='red', label='The red data')
orange_patch = mpatches.Patch(color='orange', label='The orange data')
gray_patch = mpatches.Patch(color='gray', label='The gray data')
green_patch = mpatches.Patch(color='green', label='The green data')
plt.legend([blue_patch,red_patch,orange_patch,gray_patch,green_patch],
           ["aquatic","cooking","desert","flight","vegetarian"])
plt.show()