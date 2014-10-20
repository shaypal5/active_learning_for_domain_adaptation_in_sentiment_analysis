# -*- coding: utf-8 -*-
"""
Created on Sat Oct 18 17:15:17 2014
Query By Committee sample selector: each classifier in the committee is trained on part of the data

@author: Inbar
"""
from QueryByCommitteeSampleSelector import QueryByCommiteeSampleSelector as QBC
from UncertaintySampleSelector import UncertaintySampleSelector
#from NewSampleSelector import SampleSelector
from sklearn.svm import LinearSVC

class QueryByPartialDataCommiteeSampleSelector(QBC):
    
    def __init__(self, sourceClassifier):
        QBC.__init__(self, sourceClassifier)
    
    '''
    samplesPool: a pool of samples to select from
    batchSize: number of samples to select
    '''
    def selectSamples(self, currTargetClassifier, samplesPool, batchSize):
        print('QBC')
        #if there is only one classifier in the committee, use UncertaintySampleSelector as selection strategy.        
        if len(self.committee) < 2:
            uncertaintySelector = UncertaintySampleSelector()
            samplesAndIndices = uncertaintySelector.selectSamples(currTargetClassifier, samplesPool, batchSize)
            #print(type(self.committee))
        else:
            samplesAndIndices = self.selectControvercialSamples(samplesPool, batchSize, currTargetClassifier)
        samples = samplesAndIndices[0] #samples are the instances and labels - [(x1, x2...), (y1, y2,...)]
        self.committee.append(self.trainNewClassifier(samples)) #train a new classifier and add to the committe
        return samplesAndIndices
        
    def trainNewClassifier(self, trainData):
        print('new classifier to committee')
        #trainData are the instances and labels - [(x1, x2...), (y1, y2,...)]
        X = trainData[0]
        Y = trainData[1]
        print("number of samples for new classifier is {0}".format(len(Y)))
        classifier = LinearSVC()
        classifier.fit(X,Y)
        return classifier
    