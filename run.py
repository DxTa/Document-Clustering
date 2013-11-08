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
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.decomposition import RandomizedPCA
from random import randint
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk, nltk.data, nltk.tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from difflib import get_close_matches as gcm
from itertools import chain

comparative_irregular = {
        "worse": "bad",
        "further": "far",
        "farther": "far",
        "better": "good",
        "hinder": "hind",
        "lesser": "less",
        "less": "little",
        "more": "many"
        }

superlative_irregular = {
        "worst": "bad",
        "farthest": "far",
        "furthest": "far",
        "best": "good",
        "hindmost": "hind",
        "least": "less",
        "most": "many",
        "eldest": "old"
        }

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

FILE_DIR = "./reuters21578/"
TOPICS = [x.strip() for x in open('reuters21578/all-topics-strings.lc.txt').readlines()]
stopwords_list = stopwords.words('english')
tagger = nltk.data.load(nltk.tag._POS_TAGGER) #initiate pos_tag data
wnl = WordNetLemmatizer()

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

    sentences = nltk.sent_tokenize(text)

    iws = [] #important words
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        tokens = tagger.tag(words)
        for item in tokens:
            word = item[0]
            wl = len(word)
            pos_tag = item[1]
            if word in stopwords_list or re.match(r'\W+', word): #remove stopword and punctuation
                continue

            #word lemmatize
            if pos_tag == 'JJ': #adjective
                iws.append(word)
            elif pos_tag == 'JJR' or pos_tag == 'RBR': #comparative
                try:
                    iws.append(comparative_irregular[word])
                except KeyError:
                    if word[wl-3:wl] == 'ier':
                        iws.append(word[0:wl-3]+'y')
                    elif word[wl-2:wl] == 'er':
                        iws.append(word[0:wl-2])
                        # else donothing
            elif pos_tag == 'JJS' or pos_tag == 'RBS': #superlative
                try:
                    iws.append(superlative_irregular[word])
                except KeyError:
                    if word[wl-4:wl] == 'iest':
                        iws.append(word[0:wl-4] + 'y')
                    if word[wl-3:wl] == 'est':
                        iws.append(word[0:wl-3])
                        #else donothing
            elif pos_tag == 'NN': #noun
                iws.append(word)
            elif pos_tag == 'NNS': #noun in plural
                iws.append(wnl.lemmatize(word, 'n'))
            elif pos_tag == 'NNP': #proper noun
                iws.append(word)
            elif pos_tag == 'NNPS': #proper noun in plural
                iws.append(wnl.lemmatize(word, 'n'))
            elif pos_tag == 'RB': # adverb
                possible_adjectives = [k.name for k in chain(*[j.pertainyms() for j in chain(*[i.lemmas for i in wn.synsets(word)])])]
                if len(possible_adjectives) == 0: #for irregular adv like: correspondingly
                    if word[wl-2:wl] == 'ly':
                        iws.append(word[0:wl-2])
                else:
                    adj = gcm(word, possible_adjectives)
                    if len(adj) == 0:
                        if nltk.pos_tag([word])[0][1] == 'JJ':
                            iws.append(word)
                    else:
                        iws.append(adj[0]) #fine the most similar word
            elif pos_tag == 'SYM': #symbol
                iws.append(word)
            elif pos_tag in ['VB','VBD','VBG','VBN','VBP','VBZ']: #verb
                iws.append(wnl.lemmatize(word, 'v'))
            elif pos_tag == 'CC': #cardinal number
                iws.append(word)
            elif pos_tag == 'FW': #foreign word
                iws.append(word)
            else: #others type of pos_tag are skiped
                continue #skip
    return  ' '.join(iws)
    # return text

if __name__=='__main__':

    # initializtion
    print("initiate..")
    regex = re.compile('.+?sgm')
    filelist = [m.group(0) for m in [regex.match(l) for l in os.listdir(FILE_DIR)] if m]
    articles=[]
    articles_index=[]
    articles_true=[]

    for file in filelist[]:
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
    km = KMeans(n_clusters=true_k, init='k-means++', n_init=1, verbose=True)
    km.fit(X)
    # km_labels = km.labels_
    # km_cluster_centers = km.cluster_centers_
    # km_labels_unique = np.unique(km_labels)
    print()

    print("Evaluation:")
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
    print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
    print("Adjusted Rand-Index: %.3f" % metrics.adjusted_rand_score(labels, km.labels_))
    print()

    ###############################################################################
    # Visualize the results on PCA-reduced data
    reduced_data = PCA(n_components=2).fit_transform(X)
    km = KMeans(n_clusters=true_k, init='k-means++', n_init=1, verbose=False)
    km.fit(reduced_data)
    km_labels = km.labels_
    km_cluster_centers = km.cluster_centers_
    km_labels_unique = np.unique(km_labels)

    pl.figure(1)
    pl.clf()
    colors = []
    for i in range(true_k):
        colors.append('#%06X' % randint(0, 0xFFFFFF))
    for k, col in zip(range(true_k), colors):
        my_members = km_labels == k
        cluster_center = km_cluster_centers[k]
        pl.plot(reduced_data[my_members, 0], reduced_data[my_members, 1], 'w', markerfacecolor=col, marker='.')
        pl.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)
    pl.title('Kmeans')
    pl.xticks(())
    pl.yticks(())
    pl.show()
    # rpca=RandomizedPCA(n_components=2)
    # reduced_data = rpca.fit_transform(X)
