#!/usr/bin/env python

# install BeautifulSoup by `pip install beautifulsoup4`
from bs4 import BeautifulSoup
import os
import re
import xml.sax.saxutils as saxutils
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pylab as pl
import numpy as np
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support

FILE_DIR = "./reuters21578/"
TOPICS = [x.strip() for x in open('reuters21578/all-topics-strings.lc.txt').readlines()]

def getText(article, articles_index, truth_arr):
    soup = BeautifulSoup(article, 'xml').find('REUTERS') # read as xml --> no missing node
    article_id = soup['NEWID']
    articles_index.append(article_id)
    try:
        topic = saxutils.unescape(soup.find('TOPICS').find('D').getText())
        truth_arr.append(TOPICS.index(topic))
    except AttributeError or ValueError:
        truth_arr.append(len(TOPICS)) ##unknown topic
    # author = soup.find('AUTHOR')
    # dateline = soup.find('DATELINE')
    try:
        ttype = soup.find('TEXT')['TYPE']
        if ttype == 'UNPROC':
            text = soup.find('TEXT').getText()
        elif ttype == 'BRIEF':
            text = soup.find('TITLE').getText()
    except KeyError:
        text = soup.find('BODY').getText()
    text = saxutils.unescape(text)
    return text

# def plot(kmeans,X):
    # # kmeans.fit(X)
    # h = 0.1
    # x_min, x_max = X[:, 0].data.min() + 1, X[:, 0].data.max() - 1
    # y_min, y_max = X[:, 1].data.min() + 1, X[:, 1].data.max() - 1
    # xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # # Put the result into a color plot
    # Z = Z.reshape(xx.shape)
    # pl.figure(1)
    # pl.clf()

    # pl.plot(X[:, 0], X[:, 1], 'k.', markersize=2)
    # # Plot the centroids as a white X
    # centroids = kmeans.cluster_centers_
    # pl.scatter(centroids[:, 0], centroids[:, 1],
               # marker='x', s=20, linewidths=3,
               # color='r', zorder=10)
    # pl.title('K-means clustering on Reuters')
    # pl.xlim(x_min, x_max)
    # pl.ylim(y_min, y_max)
    # pl.xticks(())
    # pl.yticks(())
    # pl.show()

if __name__=='__main__':

    # initializtion
    regex = re.compile('.+?sgm')
    filelist = [m.group(0) for m in [regex.match(l) for l in os.listdir(FILE_DIR)] if m]
    articles=[]
    articles_index=[]
    articles_true=[]

    for file in filelist:
        fp = open(FILE_DIR + file)
        str_sgm = ''.join(fp.readlines())
        fp.close()
        articles += filter(lambda i: i.strip(), str_sgm.split('</REUTERS>'))

    vectorizer = TfidfVectorizer(min_df=1, stop_words='english', smooth_idf=True)
    X = vectorizer.fit_transform([getText(x, articles_index, articles_true) for x in articles])
    print(len(vectorizer.get_feature_names()),len(articles))
    km = KMeans(n_clusters=136, init='k-means++', n_init=1, verbose=True)
    km.fit(X)
    # print(km.labels_)
    # print(np.array(articles_true))
    # rs = precision_recall_fscore_support(np.array(articles_true), km.labels_, average='macro')
    labels=np.array(articles_true)

    print()
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
    print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
    print("Adjusted Rand-Index: %.3f" % metrics.adjusted_rand_score(labels, km.labels_))
    print()
# plot(km,X)
