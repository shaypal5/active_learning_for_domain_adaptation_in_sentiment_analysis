# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 21:34:37 2014

@author: inbar
"""

import simulateGaussianDomains as dataSimulator
import numpy as np
import collections

def getData(P0, P1, numOfSamples, P0portion):
    numOfNegSamples = round(numOfSamples * P0portion)
    numOfPosSamples = numOfSamples - numOfNegSamples
    
    posSamples = P1.getSamples(numOfPosSamples)
    negSamples = P0.getSamples(numOfNegSamples)    
    
    X = np.append(posSamples, negSamples, axis = 0)
    Ypos = [1] * numOfPosSamples ; Yneg = [0] * numOfNegSamples
    Y = Ypos + Yneg
    
    data = collections.namedtuple('data', ['X', 'Y'])
    return data(X, Y)
    

def getTrainAndTestData(P0, P1, numOfOverallSamples, trainPortion = 0.7, P0portion = 0.5):
    '''
    generate train and test data from positive distribution P1 and negative P0.
    numOfOverallSamples is the number of samples in train and test sets, from both classes
    Train is trainPortion% of numOfOverallSamples.
    P0portion setes the percentage of P0 samples from numOfOverallSamples. 
    '''
    trainSize = round(trainPortion * numOfOverallSamples)
    testSize = numOfOverallSamples - trainSize
    train = getData(P0, P1, trainSize, P0portion)
    test = getData(P0, P1, testSize, P0portion)
    
    data = collections.namedtuple('data', ['train', 'test'])
    return data(train, test)

def main():
    n = 10
    numOfSamples = 7
    
    #generate P(X|Y=1) and P(X|Y=0) for source domain
    dist = dataSimulator.generateSourceDistributions(n)
    sourceP0 = dist.P0; sourceP1 = dist.P1
    
    #check that the distribution are different enough
 #   bhCoeff = dataSimulator.getBhattacharyyaCoefficient(sourceP0, sourceP1)
 #   print(bhCoeff)
    
    #generate P(X|Y=1) and P(X|Y=0) for target domain
    dist = dataSimulator.generateTargetDistributions(sourceP0, sourceP1)
    targetP0 = dist.P0; targetP1 = dist.P1
        
    targetData = getTrainAndTestData(targetP0, targetP1, 100)
    sourceData = getTrainAndTestData(sourceP0, sourceP1, 100)
    
    
    
    
    
if __name__ == '__main__':
    main()