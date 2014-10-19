# -*- coding: utf-8 -*-
"""
Created on Sat Oct 18 17:49:48 2014

@author: Inbar
"""
import scipy.sparse as sp
import operator

class SampleSelector:
    
    def selectHighestRatedSamples(self, samplesRating, samplesPool, batchSize):
        samples = samplesPool[0]
        labels = samplesPool[1]
        sortedSamplesRating = sorted(samplesRating.items(), key=operator.itemgetter(1))        
        
        count = 0
        batch = 0 #dummy initialization
        batchLabels = []
        for i in range(len(sortedSamplesRating)):
            if count == 0:
                count += 1
                sampleTuple = sortedSamplesRating[i]
                ind = sampleTuple[0]
                #print("first ind = "+str(ind))
                batch = samples[ind]#TODO create csr_matrix
                #print(len(labels))
                batchLabels = [labels[ind]]
                indexList = [ind]
                continue
            if count < batchSize:
                sampleTuple = sortedSamplesRating[i]
                ind = sampleTuple[0]
                batch = sp.vstack([batch, samples[ind]])
                batchLabels.append(labels[ind])
                indexList.append(ind)
            else:
                break
            count += 1
            
        return [[batch,batchLabels],indexList]