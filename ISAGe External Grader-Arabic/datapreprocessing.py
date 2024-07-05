##### Data Preprocessing
import codecs as c
import numpy as np
import pandas as pd
import io
import os
import sys
from nltk import word_tokenize

from io import StringIO
import csv
from nltk import word_tokenize
import re
from math import sqrt
from tashaphyne.stemming import ArabicLightStemmer
from math import *
import linecache as cc
from collections import Counter
import string
from nltk.corpus import stopwords
import pickle
import json
import sys
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
import GlobalVar as glob

####### Get Semantic Space, StopWords,Tf-Minmax weights ... for Arabic
def generateModelsForLanguage():
    getSemanticSpace()
    generateStopWords()
    glob.pathMinMax = glob.pathMinMaxArabic
    glob.pathDictionnaire = glob.pathDictionnaireArabic

def getSemanticSpace():
        glob.data = pd.read_csv(glob.pathESArabic,header=None).__array__()
        glob.words = []
        glob.EsPath = glob.pathESArabic
        glob.EsLength = len(glob.data)-1
        glob.DictioWE = pickle.load(open(glob.pathWEArabic, 'rb'))

        with io.open(glob.pathWords, 'r', encoding="utf-8") as f:
            for i, line in enumerate(f):
                glob.words.append(line.replace("\n", ""))

def generateStopWords():
    glob.stopWords = []
    with io.open(glob.pathStopWords, encoding='utf-8') as f:
       for word in f:
           glob.stopWords.append(word.replace("\n", ""))

###Arabic cleaning data
def clean_data(sentence):
   sentence = re.sub( "[\$|£|€|a-zA-Z|:|\ufeff|\‘|َ|ُ|ْ|َِِ|ّ|ً|ٌ]","", sentence)
   sentence=re.sub("[\(|=|.|,|;|،|0-9|\)|<|>|!|?|»|«|/|\+|\*|\(\)|\-|\[|\]|\(|\)|\{|\}|_|é|ù|è|؛|–|’\|/|؛|'\|…|ـ|&|؟|%|\“|\"|—|\”|@]"," ",sentence)
   sentence = re.sub("[\n|\r]", " ", sentence)
   sentence = re.sub("[اِ|آ|إ|أ]", "ا", sentence)
   sentence = re.sub("[ة]", "ه", sentence)
   return  (sentence)

### StopWords removal
def sentenceRemoveStop(sentence):
    s = ""
    for w in word_tokenize(sentence):
        if w not in glob.stopWords:
            s = s + w + " "
    return s

### stemming data
def stemSentence(corpus, isLight=False):
    stemmer = ArabicLightStemmer()
    sentence = ""
    for a in word_tokenize(corpus):
            stemmer.light_stem(a)
            if isLight == True:
                sentence = sentence + stemmer.get_stem() + " "
            else:
                sentence = sentence + stemmer.get_root() + " "
    return (sentence)

#Stem  set of Answers ex: [['Answer1 for Question 1','Answer2 for Question 1','Answer3 for Question 1'],['Answer1 for Question 2','Answer2 for Question 2'] ...] pas besoin à ce niveau
def stemAllCorpus(AllCorpus,isLight=True):
    newAll=[]
    for i in AllCorpus:
        corpus=[]
        for j in i:
              corpus.append(stemSentence(clean_data(sentenceRemoveStop(j)),isLight=isLight))
        newAll.append(corpus)
        corpus=[]
    return newAll

#Stem set of Model Answers (exemple of set: ['Model Answer for Question 1','Model Answer for Question 2' ...])
def stemAllModel(AllCorpus,isLight=True):
    newAll=[]
    for j in AllCorpus:
            newAll.append(stemSentence(clean_data(sentenceRemoveStop(j)),isLight=isLight))
    return newAll
