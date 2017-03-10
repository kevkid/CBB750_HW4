#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 10:40:16 2017

@author: kevin
"""
import re
import nltk
import os
from IPython.core.display import display
from IPython.display import Image, display
import IPython
from nltk import Tree
from nltk.draw.util import CanvasFrame
from nltk.draw import TreeWidget
from sklearn.datasets import fetch_20newsgroups



s = "this was not a good project, and it wasn't that easy"
match = re.findall(r'(no[t|r]*\s+)|(was[n\'t]*\s)',s)
for w in match:
    print w

doc = "Fx of obesity but no fx of coronary artery diseases."
tokens = nltk.word_tokenize(doc)
#normalized_tokens = [t.lower() for t in tokens]
stems = [nltk.PorterStemmer().stem(t) for t in tokens]
tagged = nltk.pos_tag(tokens)
chunked_sentence = nltk.ne_chunk(tagged)

#os.environ['PATH'] += os.pathsep + '/usr/bin/ghostscript'
#coronary artery diseases and fx, nltk doesn't recognize any of them

chunked_sentence.draw()#this displays the sentence

#WDT
#what whatever which whichever

#5
news = fetch_20newsgroups(subset='train',
                          categories=(['sci.space']),
                          remove=('headers', 'footers', 'quotes'))

news.target_names
ltokens = [nltk.word_tokenize(doc) for doc in news.data[0:1]]
# convert list of list of tokens (ltokens) into a list of tokens
import itertools
tokens_all = list(itertools.chain.from_iterable(ltokens))
# convert list of tokens to nltk text object
x = nltk.Text(t.lower() for t in tokens_all)
dispersion = x.dispersion_plot(['moon', 'earth', 'sun', 'mars'])

#6
sentence = "Since Francisco was five years old, swimming has been his passion."
#sentence = "Dr. J. Gubler studied dengue at the CDC."
sentence = "The receiving end assistant was Kevin. He was stirring the pot."
tokens = nltk.word_tokenize(sentence)
#normalized_tokens = [t.lower() for t in tokens]
stems = [nltk.PorterStemmer().stem(t) for t in tokens]
tagged = nltk.pos_tag(tokens)
chunked_sentence = nltk.ne_chunk(tagged)

grammar = r"""
  NP: {<DT|PP><VBG><NN>}
      {<PRP|JJ|VBD><VBG><DT|NN>*}
"""
cp = nltk.RegexpParser(grammar)

# parse the sentence using the chunker

cp.parse(chunked_sentence).draw()
cp.parse(tagged).draw()

#7
import helper_ner 
news = fetch_20newsgroups(subset='train',
                          categories=(['sci.med']),
                          remove=('headers', 'footers', 'quotes'))

sentences = []
[sentences.extend(nltk.sent_tokenize(text)) for text in news.data]
tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
chunked_sentences = nltk.ne_chunk_sents(tagged_sentences, binary=False)

named_entities = helper_ner.get_all_named_entities(chunked_sentences, structure='string')
from collections import Counter
a = dict(Counter(named_entities))

#8
from sklearn.feature_extraction.text import CountVectorizer
doc = [u"Fx of obesity but no fx of coronary artery diseases."]
ngram_vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(2, 2), min_df=1)#character n-grams
tf = ngram_vectorizer.fit_transform(doc)
len(ngram_vectorizer.get_feature_names())
#we get 40
ngram_vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=1)#character n-grams
tf = ngram_vectorizer.fit_transform(doc)
len(ngram_vectorizer.get_feature_names())
#132
#What is a garden-path sentence?
#When you initially read the sentance you are lead to believe one thing but by the end of the sentence you realize you are wrong.
#Parsey can help disambiguate sentences, but the article doesn't mention this across sentences.
#The problem we are trying to solve is anaphora resolution here is an example:
#In the following example, 1) and 2) are utterances; and together, they form a discourse. 
#1) John helped Mary. 
#2) He was kind.
#we know that the he refers to John.
#Parsey can disambiguate the subject within sentences such as with garden-path sentences
#While Anna dressed, the baby played in the crib.
#it is saying that Anna dressed herself and the baby is in the crib.
#My answer is no, anaphora resolution is still an open area of research.