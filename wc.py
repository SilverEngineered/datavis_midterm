from gensim import corpora
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
def transform_format(val):
    if val == 0:
    	return 255
    else:
    	return val

def show_word_cloud(category='rec.autos',max_words=100):
	categories = [category]
	image = os.path.join('masks',category + '.png')
	vectorizer = TfidfVectorizer()
	newsgroups = fetch_20newsgroups(subset='train',categories=categories,remove=('headers', 'footers', 'quotes'))
	newsgroups_test = fetch_20newsgroups(subset='test')
	texts = list(newsgroups.data)
	wc_text = " ".join(i for i in texts)
	mask = np.array(Image.open(image).convert('L'))
	transformed_mask = np.ndarray((mask.shape[0],mask.shape[1]), np.int32)
	for i in range(len(mask)):
		transformed_mask[i] = list(map(transform_format, mask[i]))
	wordcloud = WordCloud(background_color="black",max_font_size=50, max_words=max_words,contour_color='firebrick', contour_width=3,mask=transformed_mask).generate(wc_text)
	plt.imshow(wordcloud, interpolation='bilinear')
	plt.axis("off")
	plt.show()
	wordcloud.to_file(os.path.join('clouds',category + '_wc.png'))
show_word_cloud(category='talk.politics.guns')
#show_word_cloud(category='rec.autos')