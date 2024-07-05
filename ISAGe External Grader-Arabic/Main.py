#coding=utf-8
### Python 3.7
########## Main External Grader( : Trained Model = Linear Ridge PolynomialFeatures d° 2 ###########

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import codecs as c
import numpy as np
import pandas as pd
import io
import os
import sys
import xml.etree.ElementTree as etree
from nltk import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from io import StringIO
import csv
from nltk import word_tokenize
import re
# from sklearn.metrics import mean_squared_error
from math import sqrt
from tashaphyne.stemming import ArabicLightStemmer
from scipy.stats.stats import pearsonr
from math import *
import linecache as cc
from collections import Counter
import string
from nltk.corpus import stopwords
import pickle
from sklearn.preprocessing import PolynomialFeatures
import json
import sys
import datapreprocessing as prep
import GlobalVar as glob
import DomainKnowledge
import AnswerStatistics
import WordWeighting
import GapQuestion

#this function predicts the student grade included in [0..5] or a similarity betwenn 0 and 1
def getScore(Question,ReferenceAnswer,StudentAnswer,difficultyQuestion):
    
    #Load the specific domain model(Semantic Space) and the General domain model(Word Embeddings)
    prep.generateModelsForLanguage()
    #Stemming Answers
    StudentAnswerH = prep.stemAllCorpus(StudentAnswer, isLight=False)           # Heavy stem
    ReferenceAnswerH = prep.stemAllModel(ReferenceAnswer, isLight=False)        # Heavy stem

    StudentAnswerL = prep.stemAllCorpus(StudentAnswer, isLight=True)            # light stem
    ReferenceAnswerL = prep.stemAllModel(ReferenceAnswer, isLight=True)       # light stem

    #######################  Features Extraction   ################"

    ########### 1. Answer Length statistics features#########
    ##### Redundancy frequency
    AllQuestionRedondanceFreq = AnswerStatistics.getRedondanceFreqResponses(StudentAnswer)
    ## print("AllQuestionRedondanceFreq", AllQuestionRedondanceFreq )

    ##### Length Difference between Reference(model) answer and the student answer
    AllQuestionDiffLength = AnswerStatistics.getResPonsesDiffLength(StudentAnswer, ReferenceAnswer)
    ## print("AllQuestionDiffLength", AllQuestionDiffLength)

    ##### student answer's Length
    AllQuestionLength = AnswerStatistics.getResPonsesLength(StudentAnswer)
    ### print("AllQuestionLength", AllQuestionLength)

    ########### 2. Lexical Similarities Features #########

    #### Order Similarity
    print("ORDER SIMILARITY:")
    AllQuestionSimOrder = []
    for i in range(len(StudentAnswerH)):
        simES = AnswerStatistics.ss.SimOrderForQuestion(StudentAnswerH[i], ReferenceAnswerH[i])
        AllQuestionSimOrder.append(simES)
        print(simES)

    #### Jaccard Similarity
    print("JACCARD SIMILARITY:")
    AllQuestionSimJacc = []
    for i in range(len(StudentAnswerH)):
        simES = AnswerStatistics.ss.SimJaccardForQuestion(StudentAnswerH[i], ReferenceAnswerH[i])
        AllQuestionSimJacc.append(simES)
        print(simES)

    #### Dice Similarity
    print("DICE SIMILARITY:")
    AllQuestionSimDice = []
    for i in range(len(StudentAnswerH)):
        simES = AnswerStatistics.ss.SimDiceForQuestion(StudentAnswerH[i], ReferenceAnswerH[i])
        AllQuestionSimDice.append(simES)
        print(simES)

    #### Jaro Similarity ..
    print("JARO SIM:")
    AllQuestionSimJaro = []
    for i in range(len(StudentAnswerH)):
        sim = AnswerStatistics.ss.SimJaroForQuestion(StudentAnswerH[i], ReferenceAnswerH[i])
        AllQuestionSimJaro.append(sim)
        print(sim)

    #### STS Similarity
    print("STS SIM:")
    AllQuestionSimSTS = []
    for i in range(len(StudentAnswerH)):
        sim = AnswerStatistics.ss.SimStsForQuestion(StudentAnswerH[i], ReferenceAnswerH[i])
        AllQuestionSimSTS.append(sim)
        print(sim)

    ###### 3. Semantic Space Model for learning domain-specific features combined with Word Weighting features #########

    #### Cosine(ModelAnswer,StudentAnswer) using Semantic Space
    AllQuestionSimES = []
    for i in range(len(StudentAnswerL)):
        simES = DomainKnowledge.CosineSimForQuestionES(StudentAnswerL[i], ReferenceAnswerL[i])
        AllQuestionSimES.append(simES)
        print("Cosine(ModelAnswer,StudentAnswer) using SemanticSpace", simES)
        
    # Cosine(ModelAnswer,StudentAnswer) using Semantic Space with POS Tagging Ponderation
    AllQuestionSimPos = []
    for i in range(len(StudentAnswerL)):
        simES = WordWeighting.CosineSimForQuestionES_WithPonderation(StudentAnswerL[i], ReferenceAnswerL[i])
        AllQuestionSimPos.append(simES)
        print("Cosine(ModelAnswer,StudentAnswer) using SemanticSpace with PosTag Ponderation", simES)
        
    # Cosine(ModelAnswer,StudentAnswer) using Semantic Space with NTFLog Ponderation [NTFlog (w) =TFlog (w)/Max (TFlog)] named here as TFMINMAX
    AllQuestionSimES_TFMINMAX = []
    for i in range(len(StudentAnswerL)):
        simES = WordWeighting.getSimForQuestionMinMaxPond(StudentAnswerL[i], ReferenceAnswerL[i])
        AllQuestionSimES_TFMINMAX.append(simES)
        print("Cosine(ModelAnswer,StudentAnswer) using SemanticSpace with TF-MinMax Ponderation", simES)

    ####### 4. Word Embeddings Model for learning domain-general knowledge Text Similarity Features combined with Word Weighting features ##############

    WE_ResponsesVectors = DomainKnowledge.WE_AllQuestionCorpus(StudentAnswer, glob.DictioWE)
    WE_ModelsVectors = DomainKnowledge.WE_ModelResponses(ReferenceAnswer, glob.DictioWE)

    #### Cosine(ModelAnswer,StudentAnswer) using WE
    AllQuestionSimCosinusWE = []
    for i in range(len(WE_ResponsesVectors)):
        simCos = DomainKnowledge.Cosinus_We(WE_ResponsesVectors[i], WE_ModelsVectors[i])
        AllQuestionSimCosinusWE.append(simCos)
        print("Cosine(ModelAnswer,StudentAnswer) using WE", simCos)

    #### Cosine(ModelAnswer,StudentAnswer) using WE with TFMinMax Ponderation
    AllSimCosinusMinMaxWE = []
    for i in range(len(StudentAnswer)):  # tfminmax/ tfidfminmax
        sim = WordWeighting.getSimWEForQuestionMinMaxPond(StudentAnswer[i], ReferenceAnswer[i], glob.DictioWE, WordWeighting.getDictionnaireTFIDF(
            glob.pathDictionnaire), WordWeighting.getDictionnaireTFIDF(glob.pathMinMax))
        AllSimCosinusMinMaxWE.append(sim)
        print("Cosine(ModelAnswer,StudentAnswer) using WE with TF-MinMax Ponderation", sim)

    WE_ResponsesVectors = WordWeighting.WE_AllQuestionCorpusPos(StudentAnswer, glob.DictioWE,isPos=True)
    WE_ModelsVectors = WordWeighting.WE_ModelResponsesPos(ReferenceAnswer, glob.DictioWE,isPos=True)

    #### Cosine(ModelAnswer,StudentAnswer) using WE with POS Tagging  Ponderation
    AllQuestionSimCosinusWEPos = []
    for i in range(len(WE_ResponsesVectors)):
        simCos = DomainKnowledge.Cosinus_We(WE_ResponsesVectors[i], WE_ModelsVectors[i])
        AllQuestionSimCosinusWEPos.append(simCos)
    print(" # Cosine(ModelAnswer,StudentAnswer) using WE with PosTag Ponderation", simCos)

    ################# 5. Gap (Qestion-deviation) Features using Semantic Space
    ESExpectExprimEtoile, ESExpectExprimMoins, ESEcartInfoEtoile, ESEcartInfoMoins, ESGapPndEtoile, ESGapPndMoins = GapQuestion.GapForAllResponses(
        StudentAnswerL, ReferenceAnswerL, Question)
    print("Gap (Qestion-deviation) features using SemanticSpace:", ESExpectExprimEtoile, ESExpectExprimMoins,
          ESEcartInfoEtoile, ESEcartInfoMoins, ESGapPndEtoile, ESGapPndMoins)

    ################# 6. Gap (Qestion-deviation) Features using WE : The same features are calculated using WE vectors
    ExpectExprimEtoileWE, ExpectExprimMoinsWE, EcartInfoEtoileWE, EcartInfoMoinsWE, GapPndEtoileWE, GapPndMoinsWE = [], [], [], [], [], []
    for i in range(len(StudentAnswer)):
        gap1, gap2, gap3, gap4, gap5, gap6 = [], [], [], [], [], []
        for phrase in StudentAnswer[i]:
            WEExpectExprimEtoile, WEExpectExprimMoins, WEEcartInfoEtoile, WEEcartInfoMoins, WEGapPndEtoile, WEGapPndMoins = GapQuestion.GapCalculWE(
                phrase, ReferenceAnswer[i], "", glob.DictioWE)

            gap1.append(WEExpectExprimEtoile)
            gap2.append(WEExpectExprimMoins)

            gap3.append(WEEcartInfoEtoile)
            gap4.append(WEEcartInfoMoins)

            gap5.append(WEGapPndEtoile)
            gap6.append(WEGapPndMoins)

        ExpectExprimEtoileWE.append(gap1)
        ExpectExprimMoinsWE.append(gap2)
        EcartInfoEtoileWE.append(gap3)
        EcartInfoMoinsWE.append(gap4)
        GapPndEtoileWE.append(gap5)
        GapPndMoinsWE.append(gap6)
        
    print("#Gap (Qestion-deviation) features using WE", gap1, gap2, gap3, gap4, gap5, gap6)

    ############### Loading trained model
    model = pickle.load(open(glob.pathModelMLArabic, "rb"))
    Col_X = [AllQuestionSimSTS[0][0], AllQuestionSimJacc[0][0], AllQuestionSimDice[0][0],AllQuestionSimES_TFMINMAX[0][0], AllQuestionSimJaro[0][0],AllQuestionSimCosinusWEPos[0][0],AllQuestionSimCosinusWE[0][0],AllSimCosinusMinMaxWE[0][0],
                 AllQuestionSimES[0][0], AllQuestionSimPos[0][0],GapPndMoinsWE[0][0],EcartInfoMoinsWE[0][0],EcartInfoEtoileWE[0][0], ESExpectExprimEtoile[0][0],
                difficultyQuestion,AllQuestionLength[0][0],
                AllQuestionDiffLength[0][0], AllQuestionRedondanceFreq[0][0]]

    poly_reg = PolynomialFeatures(degree=2)
    Col_X = np.array([Col_X])
    X = poly_reg.fit_transform(Col_X)

    # Grade prediction
    grade = model.predict(X)
    grade=round(grade[0], 3)

    if grade > 5:
        grade=5
    elif grade <0:
        grade=0

    ####### score=score/5   to transform the grade to similarity
    return grade

######## Calling the grader function : an example
######## When the plugin is installed the call from the LMS is directed to the cloud and the main is run

######## The grader can be used and tested in a desktop version :
######## Introduce (question, student answer, reference answer and difficulty) to the getScore function as in the example

######## The question must be put in an  Array ex: Question=['عرف مصطلح الجريمة الإلكترونية']
Question=['عرف مصطلح الجريمة الإلكترونية']

## Reference answer must be put in an  Array ex: ReferenceAnswer  = ['هي كل سلوك غير قانوني يتم باستخدام الأجهزة الإلكترونية) الهاتف، الكمبيوتر، الانترنت (ينتج عنه حصول المجرم على فوائد مادية أو معنوية مع تحميل الضحية خسارة   وغالبا ما يكون هدف هذه الجرائم هو القرصنة من أجل سرقة أو إتلاف المعلومات وتكون عادة الانترنت أداة لها أو مسرحا لها']
ReferenceAnswer = [' هي كل سلوك غير قانوني يتم باستخدام الأجهزة الإلكترونية) الهاتف، الكمبيوتر، الانترنت (ينتج عنه حصول المجرم على فوائد مادية أو معنوية مع تحميل الضحية خسارة   وغالبا ما يكون هدف هذه الجرائم هو القرصنة من أجل سرقة أو إتلاف المعلومات وتكون عادة الانترنت أداة لها أو مسرحا لها']

#Here you must put the student answer in double array ex: StudentAnswer = [['هي سلوك غير أخلاقي يتم عن طريق وسائل الكترونية يهدف الى عائدات مادية  و يسبب اضرارا للضحية']]
#StudentAnswer  = [['هي سلوك غير أخلاقي يتم عن طريق وسائل الكترونية يهدف الى عائدات مادية  و يسبب اضرارا للضحية']]
StudentAnswer =[[' هي كل سلوك غير قانوني يتم باستخدام الأجهزة الإلكترونية) الهاتف، الكمبيوتر، الانترنت (ينتج عنه حصول المجرم على فوائد مادية أو معنوية مع تحميل الضحية خسارة   وغالبا ما يكون هدف هذه الجرائم هو القرصنة من أجل سرقة أو إتلاف المعلومات وتكون عادة الانترنت أداة لها أو مسرحا لها' ]]

#Here you must indicate the question level , int(1)=easy , int(2)=middle , int(3)=hard
difficultyQuestion = int(1)

##### call the scoring function : getScore
grade=getScore(Question,ReferenceAnswer,StudentAnswer,difficultyQuestion)
print("grade is :", grade, " (Student Answer and Reference Answer are similars at:", grade*100/5, "%)")









