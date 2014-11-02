# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 21:34:37 2014

@author: inbar
"""

import simulateGaussianDomains as dataSimulator
import numpy as np
import collections
import testActiveLearner
import math
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix

def getData(P0, P1, numOfSamples, P0portion):
    numOfNegSamples = round(numOfSamples * P0portion)
    numOfPosSamples = numOfSamples - numOfNegSamples
    
    posSamples = P1.getSamples(numOfPosSamples)
    negSamples = P0.getSamples(numOfNegSamples)
    
    X = np.append(posSamples, negSamples, axis = 0)
    print("X.size = %d" % X.size)
    print(X.shape)
    Ypos = [1] * numOfPosSamples ; Yneg = [0] * numOfNegSamples
    Y = Ypos + Yneg
    
    data = collections.namedtuple('data', ['X', 'Y'])
    return data(X, Y)
    

def getTrainAndTestData(domain, P0, P1, numOfOverallSamples, trainPortion = 0.7, P0portion = 0.5):
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
    
    domainType = collections.namedtuple('domain', ['name', 'train', 'test'])
    return domainType(domain,train, test)

def getKLdistance(P0, P1):
    cov0 = P0.getCovariance()
    mu0 = P0.getMu()
    
    cov1 = P1.getCovariance()
    mu1 = P1.getMu()
    
    d = len(cov1)
    invCov0 = np.linalg.inv(cov0)
    logDeterminant = math.log(np.linalg.det(cov0)/(np.linalg.det(cov1)))
    traceSigmas = np.trace(np.dot(invCov0, cov1))
    muDiff = mu0 - mu1
    muDiffTrans = np.transpose(muDiff)
    lastTerm = np.dot(np.dot(muDiffTrans, invCov0),muDiff)
    
    kl = 0.5 * (logDeterminant - d + traceSigmas + lastTerm)
    return kl

def ndarrayDatasetToSparseMatrices(dataset):
    newSet = []
    for inst in dataset:
        newSet.append(csr_matrix(inst))
    newSetNdarray = np.asarray(newSet)
    newSetMat = np.matrix(newSetNdarray, dtype = csr_matrix)
    newSetCsrMat = csr_matrix(newSetMat)
    return newSetMat
    
def testActiveLearnersWithToyData(sourceData, targetData):
    
    print("\n\n\n\n")
    print("Checking domain adaptation from source domain %s to target domain %s" % ('toySourceDomain', 'toyTargetDomain'))
    
    # Unpackaging train and test data for source domain
    trainXsource = sourceData.train.X
    trainYsource = sourceData.train.Y
    testXsource = sourceData.test.X
    testYsource = sourceData.test.Y
    trainSourceSize = len(trainXsource)
    vectorizer = DictVectorizer(dtype=float, sparse=True)
    encoder = LabelEncoder()
    
    # Unpackaging train and test data for target domain
    trainXtarget = targetData.train.X
    trainYtarget = targetData.train.Y
    testXtarget = targetData.test.X
    testYtarget = targetData.test.Y
    trainTargetSize = len(trainXtarget)
    print(type(trainXsource))

    '''    
    # Vectorize!
    print("\nVectorizing train sets of source and target domains.")
    vectorized = vectorizer.fit_transform(np.append(trainXsource, trainXtarget, axis = 0))
    vectorizedLabels = encoder.fit_transform(np.append(trainYsource, trainYtarget, axis = 0))
    total = trainSourceSize+trainTargetSize
    numOfFeatures = vectorized[0].get_shape()[1]
    print("Vectorizer num of features: %d" % numOfFeatures)
    
    # Split back to source and target
    newTrainXsource = vectorized[0:trainSourceSize]
    newTrainYsource = vectorizedLabels[0:trainSourceSize]
    newTrainXtarget = vectorized[trainSourceSize+1:total]
    newTrainYtarget = vectorizedLabels[trainSourceSize+1:total]
    
    # Vectorize test sets
    newTestXsource = vectorizer.transform(testXsource)
    newTestYsource = encoder.transform(testYsource)
    newTestXtarget = vectorizer.transform(testXtarget)
    newTestYtarget = encoder.transform(testYtarget)
      
    
    newTrainXsource = ndarrayDatasetToSparseMatrices(trainXsource)
    newTrainYsource = ndarrayDatasetToSparseMatrices(trainYsource)
    newTrainXtarget = ndarrayDatasetToSparseMatrices(trainXtarget)
    newTrainYtarget = ndarrayDatasetToSparseMatrices(trainYtarget)
    
    # Vectorize test sets
    newTestXsource = ndarrayDatasetToSparseMatrices(testXsource)
    newTestYsource = ndarrayDatasetToSparseMatrices(testYsource)
    newTestXtarget = ndarrayDatasetToSparseMatrices(testXtarget)
    newTestYtarget = ndarrayDatasetToSparseMatrices(testYtarget)

    '''
    
    newTrainXsource = trainXsource
    newTrainYsource = np.asarray(trainYsource)
    newTrainXtarget = trainXtarget
    newTrainYtarget = np.asarray(trainYtarget)
    newTestXsource = testXsource
    newTestYsource = np.asarray(testYsource)
    newTestXtarget = testXtarget
    newTestYtarget = np.asarray(testYtarget)
    
    # Package train and test sets
    newTrainSource = testActiveLearner.ActiveLearnerTester.dataType(newTrainXsource, newTrainYsource)
    newTestSource = testActiveLearner.ActiveLearnerTester.dataType(newTestXsource, newTestYsource)
    newTrainTarget = testActiveLearner.ActiveLearnerTester.dataType(newTrainXtarget, newTrainYtarget)
    newTestTarget = testActiveLearner.ActiveLearnerTester.dataType(newTestXtarget, newTestYtarget)
    
    # Package domains
    newSourceDomain = testActiveLearner.ActiveLearnerTester.domainType('toySourceDomain', newTrainSource, newTestSource)
    newTargetDomain = testActiveLearner.ActiveLearnerTester.domainType('toyTargetDomain', newTrainTarget, newTestTarget)
    
    #Set run parameters
    runTarget = True
    runUncertainty = True
    runPartialQBC = False
    runSTQBC = False
    batchSize = 10
    batchRange = [10,15,20]
    results = testActiveLearner.testActiveLearners(newSourceDomain, newTargetDomain, runTarget, runUncertainty, runPartialQBC, runSTQBC, batchSize, batchRange)
    return results

def main(): 
    n = 500
#    numOfSourceSamples = 1500 #train = 350
#    numOfTargetSamples = 3600 # train = 420
    numOfSourceSamples = 4000 #train = 350
    numOfTargetSamples = 4000 # train = 420
    
    #generate P(X|Y=1) and P(X|Y=0) for source domain
    dist = dataSimulator.generateSourceDistributions(n)
    sourceP0 = dist.P0; sourceP1 = dist.P1
    
    #check that the distribution are different enough
 #   bhCoeff = dataSimulator.getBhattacharyyaCoefficient(sourceP0, sourceP1)
 #   print(bhCoeff)
    alphas = [1, 0.95, 0.9, 0.85, 0.8]
    sourceAccuracy = []
    targetAccuracy = []
    uncertaintyAccuracy = []
    KL = []
    
    for alpha in alphas:
        print(alpha)
        #generate P(X|Y=1) and P(X|Y=0) for target domain
        dist = dataSimulator.generateTargetDistributions(sourceP0, sourceP1,alpha)
        targetP0 = dist.P0; targetP1 = dist.P1
        
        KL0 = getKLdistance(sourceP0, targetP0)
        KL1 = getKLdistance(sourceP1, targetP1)
        print("KL0: {0}, KL1: {1}".format(KL0, KL1))
        print("KL: {0}".format((KL0 + KL1)/2))
        KL.append((KL0 + KL1)/2)
            
        targetData = getTrainAndTestData('target', targetP0, targetP1, numOfTargetSamples)
        sourceData = getTrainAndTestData('source', sourceP0, sourceP1, numOfSourceSamples)
        
        results = testActiveLearnersWithToyData(sourceData, targetData)
        sourceAccuracy.append(results.source.accuracy)
        targetAccuracy.append(results.target.accuracy)
        uncertaintyAccuracy.append(results.uncertainty.accuracy)
    print(sourceAccuracy.append(results.source.accuracy))
    
if __name__ == '__main__':
    main()