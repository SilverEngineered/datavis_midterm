from gensim.models.ldamodel import LdaModel
from sklearn.datasets import fetch_20newsgroups
from gensim import corpora
import pyLDAvis.gensim
import pandas as pd



documents = fetch_20newsgroups(remove=('headers','footers')).data
texts = [[word for word in document.lower().split()] for document in documents]
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]


lda_model = LdaModel(corpus=corpus, id2word = dictionary, num_topics=20)
vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary=lda_model.id2word)
pyLDAvis.save_html(vis, 'LDA_Visualization.html')