# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 14:26:42 2014

@author: inbar
"""

class WordFrequencyModel:
    
    def __init__(self):
        self.totalInstancesProccessed = 0
        self.totalPosWords = 0
        self.totalNegWords = 0
        self.averageLineLength = 0
        self.totalPosFreq = {}
        self.totalNegFreq = {}

    def processLine(self, line, isPositiveLine):
        if isPositiveLine:
            totalFreq = self.totalPosFreq
        else:
            totalFreq = self.totalNegFreq
            
        total = 0
            
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
        
        for word in self.totalNegFreq :
            self.totalNegFreq[word] = self.totalNegFreq[word] / self.totalNegWords
            
        self.averageLineLength = (self.totalPosWords+self.totalNegWords) / len(X)
            
    def printModelFrequencies(self):
        
        for word in self.totalPosFreq :
            print("%s : %0.8f" % (word,self.totalPosFreq[word]))
        
        for word in self.totalNegFreq :
            print("%s : %0.8f" % (word,self.totalNegFreq[word]))
        
        print("total instances proccessed: %d" % self.totalInstancesProccessed)
        
        print("average line length: %0.3f" % self.averageLineLength)