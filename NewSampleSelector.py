# -*- coding: utf-8 -*-
"""
Created on Sat Oct 18 17:49:48 2014

@author: Inbar
"""
import numpy as np
import scipy.sparse as sp
import operator
import random
#from UncertaintySampleSelector import UncertaintySampleSelector

class SampleSelector:
    SCORE = 1
    SAMPLE_INDEX = 0
    
    def __init__(self):
        self.randomTieBreaker = 1
    
    def robustAppend(self, batch, toAdd):
        print("In SampleSelector.robustAppend")
        if type(batch) == sp.csr_matrix:
            return sp.vstack([batch, toAdd])
        elif type(batch) == np.ndarray:
            print("In SampleSelector.robustAppend with batch.shape[0] = %d and toAdd.shape[0] = %d" % (batch.shape[0],toAdd.shape[0]))
            print(batch.shape)
            print(toAdd.shape)
            #print(toAdd)
            return np.append(batch, np.array([toAdd]), axis = 0)
        else:
            raise ValueError("Unsupported data input of type %s." % type(batch))
    
    def selectHighestRatedSamples(self, samplesRating, samplesPool, batchSize, currClassifier):
        #print("new sample selector!")
        samples = samplesPool[0]
        labels = samplesPool[1]
        sortedSamplesRating = sorted(samplesRating.items(), key=operator.itemgetter(1))        
        
        bestScoreIndices = []
        sampleTuple = sortedSamplesRating[0]
        bestAvailableScore = sampleTuple[self.SCORE]
        ind = sampleTuple[self.SAMPLE_INDEX]
        if type(samples) == sp.csr_matrix:
            batch = samples[ind] #taking the first sample by default
        elif type(samples) == np.ndarray:
            batch = np.array([samples[ind]]) #taking the first sample by default
        else:
            raise ValueError("Unsupported data input of type %s." % type(samples))
        batchLabels = [labels[ind]]
        indexList = [ind]

        for i in range(1,len(sortedSamplesRating)):
            sampleTuple = sortedSamplesRating[i]
            sampleScore = sampleTuple[self.SCORE]
            ind = sampleTuple[self.SAMPLE_INDEX]
            if sampleScore <= bestAvailableScore: #reminder: the smaller the score, the better
                bestScoreIndices.append(ind)
            else:
                result = self.addSamplesToBatch(batch, batchLabels, bestScoreIndices, samplesPool, batchSize, indexList, currClassifier)
                batch = result[0]
                batchLabels = result[1]
                indexList = result[2]
                if len(batchLabels) == batchSize:
                    break #we have enough samples
                bestAvailableScore = sampleScore
                bestScoreIndices = [ind]
        
        print("selectHighestRatedSamples returns type(batch) = %s" %type(batch))
        return [[batch,batchLabels],indexList]
    
    def addSamplesToBatch(self, batch, batchLabels, bestScoreIndices, samplesPool, batchSize, indexList, currClassifier):
        samples = samplesPool[0]
        labels = samplesPool[1]
        numOfNeededSamples = batchSize - len(batchLabels)
        if numOfNeededSamples <= 0:
            return [batch, batchLabels, indexList]
        elif len(bestScoreIndices) <= numOfNeededSamples:
            #add all samples
            for i in range(len(bestScoreIndices)):
                ind = bestScoreIndices[i]
                batch = self.robustAppend(batch,samples[ind])                    
                batchLabels.append(labels[ind])
                indexList.append(ind)            
        else:
            #add numOfNeededSamples
            if self.randomTieBreaker:
                randomIndices = random.sample(set(bestScoreIndices), numOfNeededSamples)
                for i in range(len(randomIndices)):
                    ind = randomIndices[i]
                    #batch = sp.vstack([batch, samples[ind]])
                    batch = self.robustAppend(batch,samples[ind])  
                    batchLabels.append(labels[ind])
                    indexList.append(ind)
            else:
                print('not random!!!!!!!!!!!!!!1')
                ind = bestScoreIndices[0]
                if type(samples) == sp.csr_matrix:
                    tmpBatch = samples[0] #taking the first sample by default
                elif type(samples) == np.ndarray:
                    tmpBatch = np.array([samples[0]]) #taking the first sample by default
                else:
                    raise ValueError("Unsupported data input of type %s." % type(samples))
                tmpBatchLabels = [labels[ind]]
                indexMapping = {}
                for i in range(1, len(bestScoreIndices)):
                    ind = bestScoreIndices[i]
                    #tmpBatch = sp.vstack([batch, samples[ind]])                    
                    tmpBatch = self.robustAppend(batch,samples[ind]) 
                    tmpBatchLabels.append(labels[ind])
                    indexMapping[i] = ind
                result = UncertaintySampleSelector.selectSamples(currClassifier,[tmpBatch, tmpBatchLabels],numOfNeededSamples)
                newIndices = result[1]
                for i in range(len(newIndices)):
                   ind = indexMapping[newIndices[i]]
                   #batch = sp.vstack([batch, samples[ind]])
                   batch = self.robustAppend(batch,samples[ind])  
                   batchLabels.append(labels[ind])
                   indexList.append(ind)                               
        print("addSamplesToBatch returns type(batch) = %s" %type(batch))
        return [batch, batchLabels, indexList]