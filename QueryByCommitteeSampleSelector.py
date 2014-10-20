# -*- coding: utf-8 -*-
"""
Created on Sun Oct 19 18:28:00 2014

@author: Inbar
"""

from NewSampleSelector import SampleSelector

class QueryByCommiteeSampleSelector(SampleSelector):
    
    def __init__(self, sourceClassifier):
        SampleSelector.__init__(self)
        self.committee = [sourceClassifier]
        SampleSelector.randomTieBreak = 0
        
    def selectControvercialSamples(self, samplesPool, batchSize, currClassifier):
        samples = samplesPool[0] #samples: X
        
        disagreementsDict = {}
        for i in range(len(samplesPool[1])):
            agreementsScore = self.getAgreementsScoreForSample(samples[i])
            disagreementsDict[i] = agreementsScore
                        
        return self.selectHighestRatedSamples(disagreementsDict, samplesPool, batchSize, currClassifier)   
        
    def getAgreementsScoreForSample(self, sample):
        numOfPositivePredictions = 0
        numOfNegativePredictions = 0
        #the smaller the score, the more controvertial the sample is
        for classifier in self.committee:
            perdiction = classifier.predict(sample)
            if perdiction == 1:
                numOfPositivePredictions += 1
            else:
                numOfNegativePredictions += 1
        
        score = 1 - min(numOfPositivePredictions/len(self.committee), numOfNegativePredictions/len(self.committee))
        return score