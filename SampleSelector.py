# -*- coding: utf-8 -*-
"""
Created on Sat Oct 18 17:49:48 2014

@author: Inbar
"""
import scipy.sparse as sp
import numpy as np
import operator

class SampleSelector:
    
    def robustAppend(self, batch, toAdd):
        print("In SampleSelector.robustAppend")
        if type(batch) == sp.csr_matrix:
            return sp.vstack([batch, toAdd])
        elif type(batch) == np.ndarray:
            print("In SampleSelector.robustAppend with batch.shape[0] = %d and toAdd.shape[0] = %d" % (batch.shape[0],toAdd.shape[0]))
            print(batch.shape)
            print(toAdd.shape)
            print(toAdd)
            return np.append(batch, toAdd, axis = 0)
        else:
            raise ValueError("Unsupported data input of type %s." % type(batch))
    
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
                print("type(ind) = %s" % type(ind))
                #print("first ind = "+str(ind))
                batch = samples[ind]#TODO create csr_matrix
                #print(len(labels))
                batchLabels = [labels[ind]]
                indexList = [ind]
                continue
            if count < batchSize:
                sampleTuple = sortedSamplesRating[i]
                ind = sampleTuple[0]
                #batch = sp.vstack([batch, samples[ind]])
                batch = self.robustAppend(batch, samples[ind])
                batchLabels.append(labels[ind])
                indexList.append(ind)
            else:
                break
            count += 1
            
        return [[batch,batchLabels],indexList]