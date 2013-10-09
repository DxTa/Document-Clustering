#!/usr/bin/env python

# install BeautifulSoup by `pip install beautifulsoup4`
from bs4 import BeautifulSoup
import os
import re
import xml.sax.saxutils as saxutils
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from difflib import get_close_matches as gcm
from itertools import chain

FILE_DIR = "./reuters21578/"

regex = re.compile('.+?sgm')
filelist = [m.group(0) for m in [regex.match(l) for l in os.listdir(FILE_DIR)] if m]

for file in filelist:
    str = ''.join(open(FILE_DIR + file).readlines())
    soup = BeautifulSoup(str)
    close(FILE_DIR + file)

    articles = soup.findAll('REUTERS')

    for article in articles:
        id = article['newid']
        # author = article.find('author')
        # dateline = article.find('dateline')
        # some article only have title, this is breif article title is considered as body in this case
        title = article.find('title')
        body = article.find('body')

        if body:
            text = saxutils.unescape(body)
        else:
            text = eaxutils.unescape(title)

        #remove punctuation
        words = re.findall(r'\w+', text.lower(), flags = re.UNICODE | re.LOCALE)
        #stopword filter
        important_words = [x for x in words if x not in  stopwords.words('english')]

        wnl = WordNetLemmatizer()
        for word in important_words:
            pos_tag = nltk.pos_tag([word])
            if pos_tag == 'JJ': #adjective
                word
            elif pos_tag == 'JJR': #comparative
                word
            elif pos_tag == 'JJS': #superlative
                word
            elif pos_tag == 'MD': #modal
                word
            elif pos_tag == 'NN': #noun
                word
            elif pos_tag == 'NNS': #noun in plural
                word = wnl.lemmatize(word, 'n')
            elif pos_tag == 'NNP': #proper noun
                word
            elif pos_tag == 'NNPS': #proper noun in plural
                word = wnl.lemmatize(word, 'n')
            elif pos_tag == 'RB': # adverb
                possible_adjectives = [k.name for k in chain(*[j.pertainyms() for j in chain(*[i.lemmas for i in wn.synsets(word)])])]
                if len(possible_adjectives) == 0: #for irregular adv like: correspondingly
                    if word[len(word)-2:len(word)] == 'ly'
                        word = word[0:len(word)-2]
                else:
                    word = gcm(word, possible_adjectives)[0]
            elif pos_tag == 'RBR': #adverb, comparative
                word
            elif pos_tag == 'RBS': #adverb, superlative
                word
            elif pos_tag == 'SYM': #symbol
                word
            elif pos_tag in ['VB','VBD','VBG','VBN','VBP','VBZ']: #verb
                word = wnl.lemmatize(word, 'v')
            else:
                word #discard


    # processing here

    # write result to file

