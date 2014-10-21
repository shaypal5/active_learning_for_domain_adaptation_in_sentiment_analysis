# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 17:01:10 2014

@author: inbar
"""

import sys
from sklearn.svm import LinearSVC
import scipy
import scipy.sparse as sps
import numpy

class ActiveLearner:
    
    NUM_OF_ITERATIONS_CONDITION = 1
    CLASSIFICATION_IMPROVEMENT_CONDITION = 2    
    DEFAULT_NUM_OF_ITERATIONS = 10
    DEFAULT_BATCH_SIZE = 25
    
    def __init__(self, sampleSelector, maxIterations = None, stoppingCondition = None, batchSize = None): 
        #checking for unassigned optinal arguments and assigning defaults
        if maxIterations is None:
            maxIterations = self.DEFAULT_NUM_OF_ITERATIONS
        if stoppingCondition is None:
            stoppingCondition = self.NUM_OF_ITERATIONS_CONDITION
        if batchSize is None:
            batchSize = self.DEFAULT_BATCH_SIZE
            
        #initializing fields
        self.sampleSelector = sampleSelector
        self.stoppingCondition = stoppingCondition
        self.max_num_of_iterations = maxIterations
        self.batch_size = batchSize
        
    def train(self, sourceClassifier, sourceTrainData, targetTrainData):        
        targetClassifier = sourceClassifier

        improvement = sys.maxsize
        i = 0
        unusedTargetData = targetTrainData
        targetTrainData = sourceTrainData #use all the source train data as well
        targetTrainData[1] = targetTrainData[1].tolist()
        while not self.isStoppingConditionMet(self.stoppingCondition, i, improvement):
            #print("iteration number "+str(i))
            result = self.sampleSelector.selectSamples(targetClassifier,unusedTargetData,self.batch_size)
            selectedSamples = result[0]
            selectedIndices = result[1]
 #           if firstIteration:
 #               print("in first iteration!")
 #               firstIteration = 0
 #               targetTrainData = [selectedSamples[0], selectedSamples[1]]
 #           else:
            targetTrainData[0] = scipy.sparse.vstack([targetTrainData[0], selectedSamples[0]])
            targetTrainData[1] = targetTrainData[1] + selectedSamples[1]
            unusedTargetData = self.getNewUnusedData(unusedTargetData,selectedIndices)
            targetClassifier = LinearSVC()
            targetClassifier.fit(targetTrainData[0],targetTrainData[1])
            i += 1
        
        print("active learner was trained on {0} labeled instances.".format(self.DEFAULT_BATCH_SIZE*self.DEFAULT_NUM_OF_ITERATIONS))
        return targetClassifier
        
    def getNewUnusedData(self, unused, selectedIndices):
        unusedInst = unused[0]
        unusedLabels = unused[1]
        ndarr = unusedInst.toarray()
        ndarr = numpy.delete(ndarr,selectedIndices,axis=0)
        newUnusedInst = sps.csr_matrix(ndarr)
        newUnusedLabels = [unusedLabels[i] for i in range(len(unusedLabels)) if i not in selectedIndices]
        return [newUnusedInst,newUnusedLabels]
            
    def isStoppingConditionMet(self, stoppingCondition, i, improvement):
        if stoppingCondition == self.NUM_OF_ITERATIONS_CONDITION:
            return (i == self.max_num_of_iterations)
        else:
            print('ilegal stopping condition. Using max number of iterations instead')
            return (i == self.max_num_of_iterations)
                