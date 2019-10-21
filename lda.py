from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import LatentDirichletAllocation

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print ("Topic: " + str(topic_idx))
        string = " ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])
        print(string)

dataset = fetch_20newsgroups()
documents = dataset.data

no_features = 1000


tf_vectorizer = CountVectorizer(stop_words='english')
tf = tf_vectorizer.fit_transform(documents)
tf_feature_names = tf_vectorizer.get_feature_names()

no_topics = 20

# Run LDA
lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5).fit(tf)

no_top_words = 10
display_topics(lda, tf_feature_names, no_top_words)