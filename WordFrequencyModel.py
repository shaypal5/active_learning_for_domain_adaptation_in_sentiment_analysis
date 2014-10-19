# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 14:26:42 2014

@author: inbar
"""

import matplotlib.pyplot as plt
from scipy import stats
import msvcrt

class WordFrequencyModel:
    
    def __init__(self):
        self.totalInstancesProccessed = 0
        self.totalPosWords = 0
        self.totalNegWords = 0
        self.averageLineLength = 0
        self.totalPosFreq = {}
        self.totalNegFreq = {}
        self.lineLengthDict = {}

    def processLine(self, line, isPositiveLine):
        if isPositiveLine:
            totalFreq = self.totalPosFreq
        else:
            totalFreq = self.totalNegFreq
            
        total = 0
        
        if len(line) in self.lineLengthDict:
            self.lineLengthDict[len(line)] += 1
        else:
            self.lineLengthDict[len(line)] = 1
            
        for key in line:            
            total += line[key]
            if key in totalFreq:
                totalFreq[key] = totalFreq[key] + line[key]
            else:
                totalFreq[key] = line[key]
                
        if isPositiveLine:
            self.totalPosWords += total
        else:
            self.totalNegWords += total
        
        self.totalInstancesProccessed += 1
    
    def processDomain(self, X, Y):
        for i in range(len(X)):
            self.processLine(X[i],Y[i]==1)
        
        for word in self.totalPosFreq :
            self.totalPosFreq[word] = self.totalPosFreq[word] / self.totalPosWords
            
            
        self.posTokenizer = dict(zip(list(range(len(self.totalPosFreq.keys()))), list(self.totalPosFreq.keys())))            
        self.posDist = stats.rv_discrete(name='positiveDist', values=(list(range(len(self.totalPosFreq.keys()))), list(self.totalPosFreq.values())))
        
        for word in self.totalNegFreq :
            self.totalNegFreq[word] = self.totalNegFreq[word] / self.totalNegWords
        
        self.negTokenizer = dict(zip(list(range(len(self.totalNegFreq.keys()))), list(self.totalNegFreq.keys())))            
        self.negDist = stats.rv_discrete(name='negativeDist', values=(list(range(len(self.totalNegFreq.keys()))), list(self.totalNegFreq.values())))
            
        self.averageLineLength = (self.totalPosWords+self.totalNegWords) / len(X)
        
        for length in self.lineLengthDict:
            self.lineLengthDict[length] = self.lineLengthDict[length] / len(X)
            
        self.lineLengthDist = stats.rv_discrete(name='lineLengthDist', values=(list(self.lineLengthDict.keys()), list(self.lineLengthDict.values())))
            
    def printModelFrequencies(self):
        
        for word in self.totalPosFreq :
            print("%s : %0.8f" % (word,self.totalPosFreq[word]))
        
        for word in self.totalNegFreq :
            print("%s : %0.8f" % (word,self.totalNegFreq[word]))
            
    def generateInstance(self):
        instLength = self.lineLengthDist.rvs(size=1)[0]
        newInst = {}
        isLabelPositive = np.random.randint(0, 2, size=1)
        for i in range(instLength):
            randomToken = 
            
    def printModelDetails(self):
        
        #self.printModelFrequencies()
            
        print("Line length dist:")
        #print(self.lineLengthDict)
        lineLengthFig = plt.figure(1)
        lineLengthFig.suptitle('Line Length Distribution', fontsize=30)
        plt.plot(list(self.lineLengthDict.keys()), self.lineLengthDist.pmf(list(self.lineLengthDict.keys())))
        
        print("Positive instances dist:")
        a = self.posDist.rvs(size=1)
        print(a)
        print(self.posTokenizer[a[0]])
        positiveWordFig = plt.figure(2)
        positiveWordFig.suptitle('Positive Word Distribution', fontsize=30)
        plt.plot(list(range(len(self.totalPosFreq.keys()))), self.posDist.pmf(list(range(len(self.totalPosFreq.keys())))))
       
        
        print("Negative instances dist:")
        negativeWordFig = plt.figure(3)       
        negativeWordFig.suptitle('Negative Word Distribution', fontsize=30) 
        plt.plot(list(range(len(self.totalNegFreq.keys()))), self.negDist.pmf(list(range(len(self.totalNegFreq.keys())))))
        
        print("total instances proccessed: %d" % self.totalInstancesProccessed)
        
        print("average line length: %0.3f" % self.averageLineLength)