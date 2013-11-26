#!/usr/bin/env python

# import ipdb #debug tool
import os, re, sklearn, scipy, sqlite3, random  # scipy and sklearn is required
from bs4 import BeautifulSoup # install BeautifulSoup by `pip install beautifulsoup4`
import xml.sax.saxutils as saxutils
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pylab as pl
import numpy as np
from sklearn import metrics # for measurement
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from nltk.corpus import stopwords
import Stemmer #download and install from http://snowball.tartarus.org/wrappers/guide.html
from mylib import Kmeans
from scipy.spatial.distance import cdist  # $scipy/spatial/distance.py
from scipy.sparse import issparse  # $scipy/sparse/csr.py
from scipy import sparse
import cPickle as pickle
# import memory_profiler

# np.set_printoptions(threshold=np.nan)
FILE_DIR = "./reuters21578/"
TOPICS = [x.strip() for x in open('reuters21578/all-topics-strings.lc.txt').readlines()]
stopwords_list = stopwords.words('english')
stemmer = Stemmer.Stemmer('english')
METRIC="euclidean" #define metric name
MAX_FEATURES=1000000
conn = sqlite3.connect("mydb.db") # database to store clusters and articles information
cursor = conn.cursor()
# cursor.execute('PRAGMA temp_store = MEMORY;')
cursor.execute('PRAGMA synchronous = OFF')

'''Recreate New Database'''
def createDB():
    print("recreate database...")
    conn = sqlite3.connect("mydb.db")
    cursor = conn.cursor()
    cursor.execute("begin")
    try:
        cursor.execute('''DROP TABLE clusters''')
        cursor.execute('''CREATE TABLE clusters(id)''')
    except sqlite3.OperationalError:
        cursor.execute('''CREATE TABLE clusters(id)''')

    try:
        cursor.execute('''DROP TABLE articles''')
        cursor.execute('''CREATE TABLE articles(id, author, dateline, topic, content, cluster_id)''')
    except sqlite3.OperationalError:
        cursor.execute('''CREATE TABLE articles(id, author, dateline, topic, content, cluster_id)''')
    conn.commit()

'''cdist Wrapper'''
def cdistWrapper( X, Y, p=2, **kwargs ):
    """ -> |X| x |Y| cdist array, any cdist metric
        X or Y may be sparse -- best csr
    """
        # todense row at a time, v slow if both v sparse
    sxy = 2*issparse(X) + issparse(Y)
    if sxy == 0:
        return cdist( X, Y, p=p, **kwargs )
    d = np.empty( (X.shape[0], Y.shape[0]), np.float64 )
    if sxy == 2:
        for j, x in enumerate(X):
            d[j] = cdist( x.todense(), Y, p=p, **kwargs ) [0]
    elif sxy == 1:
        for k, y in enumerate(Y):
            d[:,k] = cdist( X, y.todense(), p=p, **kwargs ) [0]
    else:
        for j, x in enumerate(X):
            for k, y in enumerate(Y):
                d[j,k] = cdist( x.todense(), y.todense(), p=p, **kwargs ) [0]
    return d


'''get content of article'''
def getText(article, truth_arr):
    soup = BeautifulSoup(article, 'xml').find('REUTERS') # read as xml --> no missing node
    article_id = soup['NEWID'] #article id
    try:
        topic = saxutils.unescape(soup.find('TOPICS').find('D').getText())
        truth_arr.append(TOPICS.index(topic))
    except AttributeError or ValueError:
        truth_arr.append(len(TOPICS)) ##unknown topic
        topic = ""
    try:
        author = soup.find('AUTHOR').getText() #get author name
    except AttributeError:
        author = ""
    try:
        dateline = soup.find('DATELINE').getText() # get dateline of article
    except AttributeError:
        dateline = ""
    try:
        ttype = soup.find('TEXT')['TYPE']
        if ttype == 'UNPROC':
            text = soup.find('TEXT').getText() # none define format, no title, no body, article content inside 'text' tag
        elif ttype == 'BRIEF':
            text = soup.find('TITLE').getText() # brief article, only have title, no body
    except KeyError:
        text = soup.find('BODY').getText() # get body of article
    text = saxutils.unescape(text) # unescape

    # commit article to database
    cursor.execute("INSERT INTO articles (id, author, dateline, topic, content) VALUES (?,?,?,?,?)", (article_id, author, dateline, topic, text))
    # conn.commit()

    #remove punctuation
    words = re.findall(r'\w+', text.lower(), flags = re.UNICODE | re.LOCALE)
    #stopword filter
    iws = [x for x in words if x not in  stopwords_list]
    return  ' '.join(stemmer.stemWords(iws)) #return stemmed words using stemmer
    # return text

'''Main processing, kmeans, visualization'''
# @profile
def doProcessing():
    # initializtion
    print("initiate..")
    regex = re.compile('.+?sgm')
    filelist = [m.group(0) for m in [regex.match(l) for l in os.listdir(FILE_DIR)] if m]
    articles=[]
    articles_true=[] #store true label of articles
    createDB() #recreate database

    for file in filelist:
        fp =  open(FILE_DIR + file)
        str_sgm = ''.join(fp.readlines())
        fp.close()
        articles += filter(lambda i: i.strip(), str_sgm.split('</REUTERS>'))
    print()

    print("Extracting features from the training dataset using a sparse vectorizer")
    vectorizer = CountVectorizer(max_df=0.7, max_features=MAX_FEATURES)
    transformer = TfidfTransformer(use_idf=True, smooth_idf=True)
    train_data = [getText(x, articles_true) for x in articles]
    trainVectorizerArray = vectorizer.fit_transform(train_data).toarray()
    transformer.fit(trainVectorizerArray)
    X = transformer.transform(trainVectorizerArray)
    del trainVectorizerArray


    fp =  open("vocabulary.txt","w") # write vacabulary to file
    fp.write(str(vectorizer.vocabulary_))
    fp.close()
    transformer.idf_.dump('idf.dat') # dump idf_ data to file

    print(len(vectorizer.get_feature_names()),len(articles)) #show all features, number of articles
    labels=np.array(articles_true)
    true_k=np.unique(labels).shape[0] # choose number of cluster as the number of cluster defined in dataset
    print()

    # print("Performing dimensionality reduction using LSA") # make kmeans run extremely fast
    # lsa = TruncatedSVD(400) #n_demensions
    # X = lsa.fit_transform(X)
    # X = Normalizer(copy=False).fit_transform(X)
    # print()

    print("No of Cluster: %s" % true_k)
    print("Kmeans....")
    km = Kmeans.Kmeans(X, k=true_k, metric=METRIC, delta=.001, max_iter=1000, verbose=0) #kmeans...
    km_cluster_centers=km.centers
    km_labels=km.labels
    km_distances=km.distances
    print()

    print("Evaluation:")
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km_labels))
    print("Completeness: %0.3f" % metrics.completeness_score(labels, km_labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels, km_labels))
    print("Adjusted Rand-Index: %.3f" % metrics.adjusted_rand_score(labels, km_labels))

    # Visualize the results on TruncatedSVD-reduced data
    print("Visualization")
    pca = TruncatedSVD(2) #n_demensions
    reduced_data = pca.fit_transform(X)
    kmv = Kmeans.Kmeans(reduced_data, k=true_k, metric=METRIC, delta=.001, max_iter=1000, verbose=0)
    kmv_cluster_centers=kmv.centers
    kmv_labels=kmv.labels
    kmv_labels_unique=np.unique(kmv_labels)

    pl.figure(1)
    pl.clf()
    colors = []
    for i in range(true_k):
        colors.append('#%06X' % random.randint(0, 0xFFEFDB)) #not random to extreme bright color, hard to see
    print()
    for k, col in zip(range(true_k), colors):
        my_members = kmv_labels == k
        cluster_center = kmv_cluster_centers[k]
        pl.plot(reduced_data[my_members, 0], reduced_data[my_members, 1], 'w', markerfacecolor=col, marker='.')
        pl.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)
    pl.title('Kmeans')
    pl.xticks(())
    pl.yticks(())
    # pl.show()
    pl.savefig('kmeans.png')
    # ipdb.set_trace()
    print()

    # storing to database
    print("storing articles cluster to database")
    with open('X.dat', 'wb') as outx:
        pickle.dump(X, outx, pickle.HIGHEST_PROTOCOL) # dump vectorized articles to file
    print("X")
    with open('km_cluster_centers.dat', 'wb') as outcc: # dump cluster centers info to file
        pickle.dump(sparse.csr_matrix(km_cluster_centers), outcc, pickle.HIGHEST_PROTOCOL)
    print("cluster_centers")
    cursor.executemany("INSERT INTO clusters values (?)", [(str(i),) for i in range(true_k)]) #populate table clusters
    conn.commit()
    print("clusters")

    for i in range(len(km_labels)): #update information: which cluster each article belong to
        cursor.execute("UPDATE articles SET cluster_id=:cluster_id  WHERE id=:id", {'cluster_id':str(km_labels[i]), 'id':str(i+1)})
    conn.commit()
    print("articles")
    print()

    return X, km, vectorizer, transformer.idf_

def printArticle(article): #tuple
    text="ID: %s\n" % article[0]
    text+="Author: %s\n" % article[1]
    text+="Dateline: %s\n" % article[2]
    text+="Topic: %s\n" % article[3]
    text+="Content: \n%s\n-----------------------\n" % article[4]
    print(text)
    return text

if __name__=='__main__':

    '''menu'''

    while(True):
        choice = raw_input("1. Create Database\n2. Clustering using Kmeans\n3. Search\n4. Exit\nChoice: ")
        if choice is "1":
            createDB()
        elif choice is "2":
            X, km, vectorizer, idf_ = doProcessing()
        elif choice is "3":
            query = raw_input("Query: ")
            if 'km' in locals(): km_centers = km.centers
            if 'vectorizer' or 'km_centers' or 'idf_' or 'X' not in locals():
                with open("vocabulary.txt") as fp:
                    vectorizer = CountVectorizer(max_df=0.7, max_features=MAX_FEATURES, vocabulary=eval(fp.readline()))
                idf_ = np.load('idf.dat')
                with open('km_cluster_centers.dat', 'rb') as incc:
                    km_centers = pickle.load(incc)
                with open('X.dat', 'rb') as inx:
                    X = pickle.load(inx)

            # query is transform into vector
            testVectorizerArray = vectorizer.transform([' '.join(re.findall(r'\w+', query.lower(), flags = re.UNICODE | re.LOCALE))]).toarray()
            queryX = sklearn.preprocessing.normalize(np.array(testVectorizerArray*idf_),norm='l2')

            D = cdistWrapper( queryX, km_centers, metric=METRIC )  #compute similarity between query and clusters
            clusterX = D.argmin() #find the most similar cluster

            rs = cursor.execute("select id from articles where cluster_id=:cluster_id", { 'cluster_id':str(clusterX)} )
            ids_index = [eval(i[0])-1 for i in rs] # index of each article in X
            D = cdistWrapper( queryX, X[ids_index,:], metric=METRIC ) #compute similarity between query and articles in cluster
            ids_sorted = [ids_index[i[0]]+1 for i in sorted(enumerate(D[0]), key=lambda x:x[1])] # sorted id of articles by similarity of articles

            rs = cursor.execute("select * from articles where id in (%s)" % ','.join('?'*len(ids_sorted)), [str(i) for i in ids_sorted])
            rs = rs.fetchall()

            fwrite = open("search_result.txt",'w')
            for i in ids_sorted[0:100]: #only show the most 100 similar articles
                text = printArticle(rs[zip(*rs)[0].index(str(i))])
                fwrite.write(text)
            fwrite.close()
        elif choice is "4":
            break

