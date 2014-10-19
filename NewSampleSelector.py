# -*- coding: utf-8 -*-
"""
Created on Sat Oct 18 17:49:48 2014

@author: Inbar
"""
import scipy.sparse as sp
import operator
import random

class SampleSelector:
    SCORE = 1
    SAMPLE_INDEX = 0
    
    def selectHighestRatedSamples(self, samplesRating, samplesPool, batchSize):
        print("new sample selector!")
        samples = samplesPool[0]
        labels = samplesPool[1]
        sortedSamplesRating = sorted(samplesRating.items(), key=operator.itemgetter(1))        
        
        bestScoreIndices = []
        sampleTuple = sortedSamplesRating[0]
        bestAvailableScore = sampleTuple[self.SCORE]
        ind = sampleTuple[self.SAMPLE_INDEX]
        batch = samples[ind] #taking the first sample by default
        batchLabels = [labels[ind]]
        indexList = [ind]

        for i in range(1,len(sortedSamplesRating)):
            sampleTuple = sortedSamplesRating[i]
            sampleScore = sampleTuple[self.SCORE]
            ind = sampleTuple[self.SAMPLE_INDEX]
            if sampleScore <= bestAvailableScore: #reminder: the smaller the score, the better
                bestScoreIndices.append(ind)
            else:
                result = self.addSamplesToBatch(batch, batchLabels, bestScoreIndices, samplesPool, batchSize, indexList)
                batch = result[0]
                batchLabels = result[1]
                indexList = result[2]
                if len(batchLabels) == batchSize:
                    break #we have enough samples
                bestAvailableScore = sampleScore
                bestScoreIndices = [ind]
            
        return [[batch,batchLabels],indexList]
    
    def addSamplesToBatch(self, batch, batchLabels, bestScoreIndices, samplesPool, batchSize, indexList):
        samples = samplesPool[0]
        labels = samplesPool[1]
        numOfNeededSamples = batchSize - len(batchLabels)
        if numOfNeededSamples <= 0:
            return [batch, batchLabels, indexList]
        elif len(bestScoreIndices) <= numOfNeededSamples:
            #add all samples
            for i in range(len(bestScoreIndices)):
                ind = bestScoreIndices[i]
                batch = sp.vstack([batch, samples[ind]])
                batchLabels.append(labels[ind])
                indexList.append(ind)            
        else:
            #add numOfNeededSamples
            randomIndices = random.sample(set(bestScoreIndices), numOfNeededSamples)
            for i in range(len(randomIndices)):
                ind = randomIndices[i]
                batch = sp.vstack([batch, samples[ind]])
                batchLabels.append(labels[ind])
                indexList.append(ind)
                
        return [batch, batchLabels, indexList]