#!/usr/bin/env python

# install BeautifulSoup by `pip install beautifulsoup4`
from bs4 import BeautifulSoup
import os
import re
import xml.sax.saxutils as saxutils
import nltk, nltk.data, nltk.tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from difflib import get_close_matches as gcm
from itertools import chain
import multiprocessing as mp

FILE_DIR = "./reuters21578/"

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

def articleProcessing(article, queue):
    soup = BeautifulSoup(article, 'xml').find('REUTERS') # read as xml --> no missing node
    article_id = soup['NEWID']
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


    #write result to file
    article_result = (article_id, iws)
    queue.put(article_result)
    print(article_id)
    return article_result

def queue_listener(q):
    '''result listener, write to file'''
    f = open(FILE_DIR + "reut2_result.txt","w")
    while 1:
        m = q.get()
        if m == 'kill':
            break
        f.write(str(m) + '\n')
        f.flush()
    f.close()


if __name__=='__main__':

    # initializtion
    regex = re.compile('.+?sgm')
    filelist = [m.group(0) for m in [regex.match(l) for l in os.listdir(FILE_DIR)] if m]
    stopwords_list = stopwords.words('english')
    tagger = nltk.data.load(nltk.tag._POS_TAGGER) #initiate pos_tag data
    wnl = WordNetLemmatizer()
    pool = mp.Pool(mp.cpu_count() + 2)
    manager = mp.Manager()
    queue = manager.Queue()
    watcher = pool.apply_async(queue_listener, (queue,))

    for file in filelist:
        fp = open(FILE_DIR + file)
        str_sgm = ''.join(fp.readlines())
        fp.close()

        articles = filter(lambda i: i.strip(), str_sgm.split('</REUTERS>'))

        # fresult = open(FILE_DIR + file.split('.')[0] + '_result.txt','w')
        jobs=[]

        for article in articles:
            job = pool.apply_async(articleProcessing, (article, queue))
            jobs.append(job)
            # pool.map(articleProcessing, [(article,fresult,lock) for article in articles])

        for t in jobs:
            t.get()


    queue.put('kill')
    pool.close()
