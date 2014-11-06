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
        self.hadNonZeroScoreYet = False
        
    
    def getSentScore(self, sample):
        count = 0
        pos = 0
        neg = 0
        nonZero = sample.nonzero()
        nonCol = nonZero[1]
        
        for i in nonCol:
            word = self.vectorizer.get_feature_names()[i]
            if '_' not in word:
                count += 1
                sentiment = self.sentimMeasure.getSentimentOfWord(word)
                if sentiment == 1:
                    pos += 1
                elif sentiment == -1:
                    neg += 1
        
        if pos == 0 or neg == 0:
            return 0
        
        polarityScore = min(pos/neg, neg/pos)
        if (polarityScore != 0) and not self.hadNonZeroScoreYet:
            self.hadNonZeroScoreYet = True
        return polarityScore
            
    
    def selectSamples(self, svm,samplesPool,batchSize):
        #print("selectSamples() in SentimentIntensitySampleSelector")
        samples = samplesPool[0]
        sent_scores = [self.getSentScore(sample) for sample in samples]
        if not self.hadNonZeroScoreYet:
            print("Only zero scores in this iteration of Sentiment Intensity selectSamples() !!!")
        #print("The sentiment scores in SentimentIntensitySampleSelector:")
        #print(sent_scores)
        scoreDict = {}
        for i in range(len(sent_scores)):
            scoreDict[i] = sent_scores[i]
        
        return self.selectHighestRatedSamples(scoreDict, samplesPool, batchSize, svm)