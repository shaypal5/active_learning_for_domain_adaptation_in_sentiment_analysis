# -*- coding: utf-8 -*-
"""
Created on Sun Oct 19 18:18:24 2014
Query By a Committee containing source classifier and target classifier

@author: Inbar
"""

from QueryByCommitteeSampleSelector import QueryByCommiteeSampleSelector as QBC
from UncertaintySampleSelector import UncertaintySampleSelector

class TargetAndSourceQBCSampleSelector(QBC):
    
    def __init__(self, sourceClassifier):
        QBC.__init__(self, sourceClassifier)
        self.firstQuery = 1
        
    '''
    samplesPool: a pool of samples to select from
    batchSize: number of samples to select
    '''
    def selectSamples(self, currTargetClassifier, samplesPool, batchSize):
        if self.firstQuery:
            uncertaintySelector = UncertaintySampleSelector()
            samplesAndIndices = uncertaintySelector.selectSamples(currTargetClassifier, samplesPool, batchSize)
            self.committee.append(currTargetClassifier)
            self.firstQuery = 0
        else:
            self.committee[1] = currTargetClassifier #update target classifier
            samplesAndIndices = self.selectControvercialSamples(samplesPool, batchSize, currTargetClassifier)
        return samplesAndIndices
        