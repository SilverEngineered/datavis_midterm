from sklearn.datasets import fetch_20newsgroups
from matplotlib import pyplot as plt

newsgroups = fetch_20newsgroups()


counts = []
for j in range(20):
    counts.append(len([i for i in newsgroups.target if i==j]))


#counts = [480, 584, 591, 590, 578, 593, 585, 594, 598, 597, 600, 595, 591, 594, 593, 599, 546, 564, 465, 377]
targets = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']

patches, texts = plt.pie(counts, labels=targets)
plt.axis('equal')
plt.show()
