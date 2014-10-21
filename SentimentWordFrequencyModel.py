# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 20:25:00 2014

@author: Shay
"""

from enum import Enum
from nltk.corpus import sentiwordnet as swn
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

class WordNetKey(Enum):
    a = 'a' #adjective
    n = 'n' #noun
    r = 'r' #rambocsious
    v = 'v' #verb
    
class SentimentWordFrequencyModel:
    
    def __init__(self):
        self.totalInstancesProccessed = 0
        self.totalPosInstances = 0
        self.totalNegInstances = 0
        
        self.totalPosWordsInDomain = 0 #total number of positive words in positive instance in the entire domain
        self.totalNegWordsInDomain = 0 #total number of negative words in negative instance in the entire domain
        self.totalObjWordsInDomain = 0 #total number of objective words in all instance in the entire domain
        
        #Frequencies of sentimentful words in positive/negative instances
        self.totalPosFreq = {}
        self.totalNegFreq = {}
        #Frequencies of objective words in all sentences
        self.totalObjFreq = {}
        
        #Line length distributions for positive and negative sentences
        self.posLineLengthDict = {}
        self.negLineLengthDict = {}
        
        #sentiment-full word percentage for positive and negative sentences
        self.sentWordPercentageInPos = 0
        self.sentWordPercentageInNeg = 0
    
    def getSentimentOfWord(self, word):
        sentSet = list(swn.senti_synsets(word))
        
        #if not found, assume objective word
        if len(sentSet) == 0:
            #print('empty sentSet for word '+word)
            return 0
        #else:
            #print('non empty sentSet for word '+word)
            
        totalPos = 0
        totalNeg = 0
        totalObj = 0
        for sentiword in sentSet:
            totalPos += sentiword.pos_score()
            totalNeg += sentiword.neg_score()
            totalObj += sentiword.obj_score()
        
        totalPos = totalPos / len(sentSet)
        totalNeg = totalNeg / len(sentSet)
        totalObj = totalObj / len(sentSet)
            
        #determine sentiment
        if (totalPos >= totalObj) and (totalPos >= totalNeg):
            return 1
        if (totalNeg >= totalObj) and (totalNeg >= totalPos):
            return -1
        if (totalObj >= totalPos) and (totalObj >= totalNeg):
            return 0
        
    def processPositiveLine(self, line):        
        if len(line) in self.posLineLengthDict:
            self.posLineLengthDict[len(line)] += 1
        else:
            self.posLineLengthDict[len(line)] = 1

        totalPosInThisLine = 0        
        
        for key in line:
            wordSent = self.getSentimentOfWord(key)
            if wordSent != -1: #if this word is not negative
                if wordSent == 0: #if this is an objective word
                    freq = self.totalObjFreq
                    self.totalObjWordsInDomain += line[key]
                if wordSent == 1: #if this is a positive word
                    freq = self.totalPosFreq
                    self.totalPosWordsInDomain += line[key]
                    totalPosInThisLine += line[key]
                
                if key in freq:
                    freq[key] = freq[key] + line[key]
                else:
                    freq[key] = line[key]
                    
        posPercentage = totalPosInThisLine / len(line)
        self.sentWordPercentageInPos += posPercentage
        
    def processNegativeLine(self, line):        
        if len(line) in self.negLineLengthDict:
            self.negLineLengthDict[len(line)] += 1
        else:
            self.negLineLengthDict[len(line)] = 1  

        totalNegInThisLine = 0        
            
        for key in line:
            wordSent = self.getSentimentOfWord(key)
            if wordSent != 1: #if this word is not positive
                if wordSent == 0: #if this is an objective word
                    freq = self.totalObjFreq
                    self.totalObjWordsInDomain += line[key]
                if wordSent == -1: #if this is a negative word
                    freq = self.totalNegFreq
                    self.totalNegWordsInDomain += line[key]
                    totalNegInThisLine += line[key]
                
                if key in freq:
                    freq[key] = freq[key] + line[key]
                else:
                    freq[key] = line[key]
                    
        negPercentage = totalNegInThisLine / len(line)
        self.sentWordPercentageInNeg += negPercentage
        

    def processLine(self, line, isPositiveLine):
        if isPositiveLine:
            self.totalPosInstances += 1
            self.processPositiveLine(line)
        else:
            self.totalNegInstances += 1
            self.processNegativeLine(line)
            
        self.totalInstancesProccessed += 1
    
    def processDomain(self, X, Y):
        for i in range(len(X)):
            self.processLine(X[i],Y[i]==1)
            
        self.sentWordPercentageInPos = self.sentWordPercentageInPos / self.totalPosInstances
        self.sentWordPercentageInNeg = self.sentWordPercentageInNeg / self.totalNegInstances
        
        #build word distribution for positive words
        for word in self.totalPosFreq :
            self.totalPosFreq[word] = self.totalPosFreq[word] / self.totalPosWordsInDomain  
        self.posTokenizer = dict(zip(list(range(len(self.totalPosFreq.keys()))), list(self.totalPosFreq.keys())))            
        self.posDist = stats.rv_discrete(name='positiveDist', values=(list(range(len(self.totalPosFreq.keys()))), list(self.totalPosFreq.values())))
        
        #build word distribution for negative words
        for word in self.totalNegFreq :
            self.totalNegFreq[word] = self.totalNegFreq[word] / self.totalNegWordsInDomain
        self.negTokenizer = dict(zip(list(range(len(self.totalNegFreq.keys()))), list(self.totalNegFreq.keys())))            
        self.negDist = stats.rv_discrete(name='negativeDist', values=(list(range(len(self.totalNegFreq.keys()))), list(self.totalNegFreq.values())))
        
        #build word distribution for objective words
        for word in self.totalObjFreq :
            self.totalObjFreq[word] = self.totalObjFreq[word] / self.totalObjWordsInDomain
        self.objTokenizer = dict(zip(list(range(len(self.totalObjFreq.keys()))), list(self.totalObjFreq.keys())))            
        self.objDist = stats.rv_discrete(name='objectiveDist', values=(list(range(len(self.totalObjFreq.keys()))), list(self.totalObjFreq.values())))
        
        #build positive line length distribution
        for length in self.posLineLengthDict:
            self.posLineLengthDict[length] = self.posLineLengthDict[length] / self.totalPosInstances
        self.posLineLengthDist = stats.rv_discrete(name='posLineLengthDist', values=(list(self.posLineLengthDict.keys()), list(self.posLineLengthDict.values())))
        
        #build negative line length distribution
        for length in self.negLineLengthDict:
            self.negLineLengthDict[length] = self.negLineLengthDict[length] / self.totalNegInstances
        self.negLineLengthDist = stats.rv_discrete(name='negLineLengthDist', values=(list(self.negLineLengthDict.keys()), list(self.negLineLengthDict.values())))
        
    #labelIsPositive = 0 for Negative label, 1 for Positive label, 2 for random
    def generateInstance(self, labelIsPositive = 2):
        if labelIsPositive == 2:
            labelIsPositive = np.random.randint(0, 2, size=1)   
        
        if labelIsPositive:
            instLength = self.posLineLengthDist.rvs(size=1)[0]
            numOfSent = int(round(self.sentWordPercentageInPos * instLength))
        else:
            instLength = self.negLineLengthDist.rvs(size=1)[0]
            numOfSent = int(round(self.sentWordPercentageInNeg * instLength))
        
        numOfObj = instLength - numOfSent            
        newInst = {}
        
        for i in range(numOfSent):
            if labelIsPositive:
                randomToken = self.posDist.rvs(size=1)[0]
                randomWord = self.posTokenizer[randomToken]
            else:
                randomToken = self.negDist.rvs(size=1)[0]
                randomWord = self.negTokenizer[randomToken]
                
            if randomWord in newInst:
                newInst[randomWord] += 1
            else:
                newInst[randomWord] = 1
        
        for i in range(numOfObj):
            randomToken = self.objDist.rvs(size=1)[0]
            randomWord = self.objTokenizer[randomToken]
                
            if randomWord in newInst:
                newInst[randomWord] += 1
            else:
                newInst[randomWord] = 1
        
        return [newInst, labelIsPositive]
        
        
    def generateDataset(self, size, positivePercentage = 0.5):
        posNum = round(size * positivePercentage)
        negNum = size - posNum
        X = []
        Y = []
            
        for i in range(posNum):
            X.append(self.generateInstance(1)[0])
            Y.append(1)
            
        for i in range(negNum):
            X.append(self.generateInstance(0)[0])
            Y.append(0)
            
        return [X,Y]