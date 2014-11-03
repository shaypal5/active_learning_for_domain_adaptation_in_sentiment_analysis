# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 18:33:19 2014

@author: Shay
"""

from UncertaintySampleSelector import UncertaintySampleSelector
from QueryByPartialDataCommiteeSampleSelector import QueryByPartialDataCommiteeSampleSelector
from TargetAndSourceQBCSampleSelector import TargetAndSourceQBCSampleSelector
from SentimentIntensitySampleSelector import SentimentIntensitySampleSelector

import ActiveLearner

import collections
from sklearn.svm import LinearSVC
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from numpy import ndarray
import numpy as np
import random

class ActiveLearnerTester:
    dataType = collections.namedtuple('data', ['X', 'Y'])
    domainType = collections.namedtuple('domain', ['name', 'train', 'test'])
    resultsType = collections.namedtuple('result', ['name', 'correct', 'incorrect', 'TP', 'FP', 'TN', 'FN', 'precision', 'recall', 'accuracy'])
    classifiersToRunType = collections.namedtuple('classifiersToRun', ['target', 'uncertainty', 'partialQBC', 'STQBC', 'sentimentIntensity'])
    bathConfigType = collections.namedtuple('batchConfig', ['batchSize', 'batchRange'])
    partialTrainConfigType = collections.namedtuple('partialTrainConfig', ['partialSourceTrain', 'partialTargetTrain', 'partialSourceTrainSize'])
    
def getSubsetByIndices(orgSet, indices):
    if type(orgSet) == sp.csr_matrix:
        return sp.csr_matrix(np.ndarray([orgSet[i] for i in indices]))
    elif type(orgSet) == np.ndarray:
        return [orgSet[i] for i in indices]
    elif type(orgSet) == list:
        return [orgSet[i] for i in indices]
    else:
        raise ValueError("Unsupported data input of type %s in getSubsetByIndices()." % type(orgSet))

def getPartialTrain(originalSetSize, trainSet, newSetSize): 
    indexList = list(range(originalSetSize))
    random.shuffle(indexList)
    indexList = indexList[:newSetSize]
    newTrain = ActiveLearnerTester.dataType(getSubsetByIndices(trainSet.X,indexList),getSubsetByIndices(trainSet.Y,indexList))
    return newTrain

def testResultantClassifier(name, classifier, testSet):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    correct = 0
    wrong = 0
    
    if type(testSet.Y) == csr_matrix:
        ysize = testSet.Y.shape[0]
    elif type(testSet.Y) == ndarray:
        ysize = testSet.Y.size
    else:
        raise ValueError("Unsupported data input.")
    
    for i in range(ysize):
        prediction = classifier.predict(testSet.X[i])
        if prediction == testSet.Y[i]:
            correct += 1
            if prediction == 1:
                TP += 1
            else:
                TN += 1
        else:
            wrong += 1
            if prediction == 1:
                FP += 1
            else:
                FN += 1
    
    print("TP: {0} FP: {1} TN: {2} FN: {3}".format(TP, FP, TN, FN))
    if (TP+FP) > 0:
        precision = TP / (TP + FP) #out of all the examples the classifier labeled as positive, what fraction were correct?
    else:
        precision = 0
    if (TP+FN) > 0:
        recall = TP / (TP + FN) #out of all the positive examples there were, what fraction did the classifier pick up?
    else:
        recall = 0
    if (TP + TN + FP + FN) > 0:
        accuracy = (TP + TN) / (TP + TN + FP + FN)
    else:
        accuracy = 0
    print("precision: {0}  recall: {1}  accuracy: {2}".format(precision, recall, accuracy))
    print("correct: {0}, wrong: {1}".format(correct, wrong))
    thisResult = ActiveLearnerTester.resultsType(name, correct, wrong, TP, FP, TN, FN, precision, recall, accuracy)
    return thisResult
    
    '''
def checkSizes(self, trainData, testData):
    trainXlen = self.getLength(trainData.X)
    sourceYlen = self.getLength(trainData.Y)
    if sourceXlen != sourceYlen:
        raise ValueError("Source train has %d samples in X but %d in Y." % (sourceXlen, sourceYlen))
    targetXlen = self.getLength(targetTrainData[0])
    targetYlen = self.getLength(targetTrainData[1])
    if targetXlen != targetYlen:
        raise ValueError("Target train has %d samples in X but %d in Y." % (targetXlen, targetYlen))
        '''

#each domain is a tuple of (name, train, test)
def testActiveLearners(sourceDomain, targetDomain, classifiersToRun = None, bathConfig = None, partialTrainConfig = None, featureSentimentDict = None):
    
    if type(sourceDomain.train.Y) == csr_matrix:
        sourceTrainSize = sourceDomain.train.Y.shape[0]
        sourceTestSize = sourceDomain.test.Y.shape[0]
        targetTrainSize = targetDomain.train.Y.shape[0]
        targetTestSize = targetDomain.test.Y.shape[0]
    elif type(sourceDomain.train.Y) == ndarray:
        sourceTrainSize = sourceDomain.train.Y.size
        sourceTestSize = sourceDomain.test.Y.size
        targetTrainSize = targetDomain.train.Y.size
        targetTestSize = targetDomain.test.Y.size
    else:
        raise ValueError("Unsupported data input of type %s." % type(sourceDomain.train.Y))
        
    print("\n\n\n")
    print("Checking domain adaptation from source domain %s to target domain %s" % (sourceDomain.name, targetDomain.name))
    print("|Source Domain: %s | Total Size: %d | Train Set Size: %d | Test Set Size: %d |" % (sourceDomain.name, sourceTrainSize+sourceTestSize, sourceTrainSize, sourceTestSize ))
    print("|Target Domain: %s | Total Size: %d | Train Set Size: %d | Test Set Size: %d |" % (targetDomain.name, targetTrainSize+targetTestSize, targetTrainSize, targetTestSize ))
    
    #Default parameters
    if classifiersToRun == None:
        classifiersToRun = ActiveLearnerTester.classifiersToRunType(True, True, False, False, False) #Runing only target and uncertainty
    if bathConfig == None:
        bathConfig = ActiveLearnerTester.bathConfigType(10,[20])
        
    partialTargetTrain = False
    #Check for partial training parameters
    if partialTrainConfig != None:
        if partialTrainConfig.partialSourceTrain:
            sourceDomain = ActiveLearnerTester.domainType(sourceDomain.name, getPartialTrain(sourceTrainSize, sourceDomain.train, partialTrainConfig.partialSourceTrainSize), sourceDomain.test)
            print("Training source SVM using %d out of a total of %d samples" %(partialTrainConfig.partialSourceTrainSize, sourceTrainSize))
        if partialTrainConfig.partialTargetTrain:    
            partialTargetTrain = True
            partialTargetTrainSize = bathConfig.batchSize * bathConfig.batchRange[0] #number of samples to use for target train    
            partialTargetTrainSet = getPartialTrain(targetTrainSize, targetDomain.train, partialTargetTrainSize)
            print("Training target SVM using %d out of a total of %d samples" %(partialTargetTrainSize, targetTrainSize)) 
        
    #train classifier on source domain
    print("\n\n\n")
    print("=================================================================================")
    print("(1) Testing Source Classifier: ")
    sourceClassifier = LinearSVC()
    sourceClassifier.fit(sourceDomain.train.X,sourceDomain.train.Y)
    print("Source classifier was trained on %d labeled instances" % sourceTrainSize)
    sourceClassRes = testResultantClassifier('source_classifier', sourceClassifier, targetDomain.test)
    print("=================================================================================")
    
    targetClassRes = None    
    uncertaintyClassRes = None    
    partialComClassRes = None    
    targetSourceQBCClassRes = None
    sentimentIntensityClassRes = None
    
    #train classifier on target domain
    if classifiersToRun.target:
        print("\n\n\n")
        print("=================================================================================")
        print("(2) Testing Target classifier: ")
        targetClassifier = LinearSVC()
        if partialTargetTrain:
            targetClassifier.fit(partialTargetTrainSet.X, partialTargetTrainSet.Y)
            print("target classifier was trained on %d labeled instances out of a total of %d" % (partialTargetTrainSize, targetTrainSize))
        else:
            targetClassifier.fit(targetDomain.train.X, targetDomain.train.Y)            
            print("target classifier was trained on %d labeled instances" % targetTrainSize)
        targetClassRes = testResultantClassifier('target_classifier', targetClassifier, targetDomain.test)
        print("=================================================================================")
    
    #test UNCERTAINTY classifier
    if classifiersToRun.uncertainty:
        print("\n\n\n")
        print("=================================================================================") 
        print("(3) Testing Active Learning classifier with UNCERTAINTY sample selector: ")
        for numOfIter in bathConfig.batchRange:
            print("With %d iterations of %d each" % (numOfIter, bathConfig.batchSize))
            selector = UncertaintySampleSelector()
            learner = ActiveLearner.ActiveLearner(selector, numOfIter, None, bathConfig.batchSize)
            uncertaintyClassifier = learner.train(sourceClassifier,[sourceDomain.train.X,sourceDomain.train.Y],[targetDomain.train.X,targetDomain.train.Y])
            uncertaintyClassRes = testResultantClassifier('uncertainty_classifier', uncertaintyClassifier, targetDomain.test)
            print("=================================================================================")

    #test PARTIAL QBC
    if classifiersToRun.partialQBC:
        print("\n\n\n")
        print("=================================================================================")   
        print("(4) Testing Active Learning classifier with *Query By Partial Data Commitee* sample selector: ") 
        for numOfIter in bathConfig.batchRange:
            print("With %d iterations of %d each" % (numOfIter, bathConfig.batchSize))
            selector = QueryByPartialDataCommiteeSampleSelector(sourceClassifier)
            learner = ActiveLearner.ActiveLearner(selector, numOfIter, None, bathConfig.batchSize)
            partialComClassifier = learner.train(sourceClassifier,[sourceDomain.train.X,sourceDomain.train.Y],[targetDomain.train.X,targetDomain.train.Y])
            partialComClassRes = testResultantClassifier('partial_committee_classifier', partialComClassifier, targetDomain.test)
            print("=================================================================================")

    #test TARGET & SOURCE QBC
    if classifiersToRun.STQBC:
        print("\n\n\n")
        print("=================================================================================") 
        print("(5) Testing Active Learning classifier with *Target & Source QBC* sample selector: ") 
        for numOfIter in bathConfig.batchRange:
            print("With %d iterations of %d each" % (numOfIter, bathConfig.batchSize))
            selector = TargetAndSourceQBCSampleSelector(sourceClassifier)
            learner = ActiveLearner.ActiveLearner(selector, numOfIter, None, bathConfig.batchSize)
            targetSourceQBCClassifier = learner.train(sourceClassifier,[sourceDomain.train.X,sourceDomain.train.Y],[targetDomain.train.X,targetDomain.train.Y])
            targetSourceQBCClassRes = testResultantClassifier('partial_committee_classifier', targetSourceQBCClassifier, targetDomain.test)
            print("=================================================================================")

    #test SENTIMENT intensity selector
    if classifiersToRun.sentimentIntensity:
        print("\n\n\n")
        print("=================================================================================") 
        print("(6) Testing Active Learning classifier with *Sentiment Intensity* sample selector: ") 
        for numOfIter in bathConfig.batchRange:
            print("With %d iterations of %d each" % (numOfIter, bathConfig.batchSize))
            selector = SentimentIntensitySampleSelector(featureSentimentDict)
            learner = ActiveLearner.ActiveLearner(selector, numOfIter, None, bathConfig.batchSize)
            sentimentIntensityClassifier = learner.train(sourceClassifier,[sourceDomain.train.X,sourceDomain.train.Y],[targetDomain.train.X,targetDomain.train.Y])
            sentimentIntensityClassRes = testResultantClassifier('partial_committee_classifier', sentimentIntensityClassifier, targetDomain.test)
            print("=================================================================================")    
    
    print("Test done")
    results = collections.namedtuple('results', ['source', 'target', 'uncertainty', 'partialQBC', 'STQBC', 'sentimentIntensity'])    
    return results(sourceClassRes, targetClassRes, uncertaintyClassRes, partialComClassRes, targetSourceQBCClassRes, sentimentIntensityClassRes)