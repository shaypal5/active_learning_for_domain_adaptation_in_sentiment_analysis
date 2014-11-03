# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 17:49:30 2014

@author: shaypalachy
"""
from NewSampleSelector import SampleSelector
from SentimentWordFrequencyModel import SentimentWordFrequencyModel

class SentimentIntensitySampleSelector(SampleSelector):

    def __init__(self, vectorizer):
        SampleSelector.__init__(self)
        self.vectorizer = vectorizer
        self.sentimMeasure = SentimentWordFrequencyModel()
    
    def getSentScore(self, sample):
        count = 0
        score = 0
        for token in sample:
            word = self.vectorizer.get_feature_names()[token]
            count += 1
            score += abs(self.sentimMeasure.getSentimentOfWord(word))
        if count == 0:
            return 0
        return score/count
            
    
    def selectSamples(self, svm,samplesPool,batchSize):
        samples = samplesPool[0]
        sent_scores = [self.getSentScore(sample) for sample in samples]
        print("The sentiment scores in SentimentIntensitySampleSelector:")
        print(sent_scores)
        scoreDict = {}
        for i in range(len(sent_scores)):
            scoreDict[i] = sent_scores[i]
        
        return self.selectHighestRatedSamples(scoreDict, samplesPool, batchSize, svm)