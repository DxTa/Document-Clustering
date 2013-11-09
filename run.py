#!/usr/bin/env python

# scipy and sklearn is required
# install BeautifulSoup by `pip install beautifulsoup4`
from bs4 import BeautifulSoup
import os
import re
import xml.sax.saxutils as saxutils
from sklearn.feature_extraction.text import TfidfVectorizer
import pylab as pl
import numpy as np
from sklearn import metrics
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
import random
from nltk.corpus import stopwords
import Stemmer #download and install from http://snowball.tartarus.org/wrappers/guide.html
from mylib import Kmeans

FILE_DIR = "./reuters21578/"
TOPICS = [x.strip() for x in open('reuters21578/all-topics-strings.lc.txt').readlines()]
stopwords_list = stopwords.words('english')
stemmer = Stemmer.Stemmer('english')

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

    # sentences = nltk.sent_tokenize(text)
    #remove punctuation
    words = re.findall(r'\w+', text.lower(), flags = re.UNICODE | re.LOCALE)
    #stopword filter
    iws = [x for x in words if x not in  stopwords.words('english')]
    return  ' '.join(stemmer.stemWords(iws))
    # return text

if __name__=='__main__':

    # initializtion
    print("initiate..")
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
    print()

    print("Extracting features from the training dataset using a sparse vectorizer")
    vectorizer = TfidfVectorizer(max_df=0.7, use_idf=True, max_features=10000)
    X = vectorizer.fit_transform([getText(x, articles_index, articles_true) for x in articles])
    print(len(vectorizer.get_feature_names()),len(articles))
    labels=np.array(articles_true)
    true_k=np.unique(labels).shape[0]
    print()

    print("Performing dimensionality reduction using LSA")
    lsa = TruncatedSVD(100) #n_demensions
    X = lsa.fit_transform(X)
    X = Normalizer(copy=False).fit_transform(X)
    print()

    print("No of Cluster: %s" % true_k)
    metric="euclidean"
    km = Kmeans.Kmeans(X, k=true_k, metric=metric, delta=.001, max_iter=1000, verbose=0)
    km_cluster_centers=km.centers
    km_labels=km.labels
    km_distances=km.distances
    print()

    print("Evaluation:")
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km_labels))
    print("Completeness: %0.3f" % metrics.completeness_score(labels, km_labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels, km_labels))
    print("Adjusted Rand-Index: %.3f" % metrics.adjusted_rand_score(labels, km_labels))
    print()

    ###############################################################################
    # Visualize the results on PCA-reduced data
    reduced_data = PCA(n_components=2).fit_transform(X)
    km = Kmeans.Kmeans(reduced_data, k=true_k, metric=metric, delta=.001, max_iter=1000, verbose=0)
    km_cluster_centers=km.centers
    km_labels=km.labels
    km_labels_unique=np.unique(km_labels)


    pl.figure(1)
    pl.clf()
    colors = []
    for i in range(true_k):
        colors.append('#%06X' % random.randint(0, 0xFFEFDB)) #not random to extreme bright color, hard to see
    for k, col in zip(range(true_k), colors):
        my_members = km_labels == k
        cluster_center = km_cluster_centers[k]
        pl.plot(reduced_data[my_members, 0], reduced_data[my_members, 1], 'w', markerfacecolor=col, marker='.')
        pl.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)
    pl.title('Kmeans')
    pl.xticks(())
    pl.yticks(())
    pl.show()
