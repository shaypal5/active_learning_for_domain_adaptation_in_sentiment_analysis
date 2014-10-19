# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 17:49:30 2014

@author: inbar
"""
from NewSampleSelector import SampleSelector

class UncertaintySampleSelector(SampleSelector):
    
    def selectSamples(self, svm,samplesPool,batchSize):
        samples = samplesPool[0]
        confidence_scores = svm.decision_function(samples)
        #print(confidence_scores)
        confDict = {}
        for i in range(len(confidence_scores)):
            confDict[i] = abs(confidence_scores[i])
        
        return self.selectHighestRatedSamples(confDict, samplesPool, batchSize)
                
        
            
    