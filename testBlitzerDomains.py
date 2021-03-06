# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 11:14:31 2014

@author: Shay
"""

import parseProcessedDataFileForScikit
import testActiveLearner
from testActiveLearner import ActiveLearnerTester
from BlitzerDatasetDomain import BlitzerDatasetDomain
#from SentimentWordFrequencyModel import SentimentWordFrequencyModel

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder

def testActiveLearnersWithBlitzerDomains(sourceDomain, targetDomain):
    
    print("\n")
    print("Checking domain adaptation from source domain %s to target domain %s" % (sourceDomain.value, targetDomain.value))
    print("|Source Domain: %s | Total Size: %d | Train Set Size: %d | Test Set Size: %d |" % (sourceDomain.value, sourceDomain.getNumOfTotalInstanceInDomain(), sourceDomain.getNumOfTrainInstanceInDomain(), sourceDomain.getNumOfTestInstanceInDomain() ))
    print("|Target Domain: %s | Total Size: %d | Train Set Size: %d | Test Set Size: %d |" % (targetDomain.value, targetDomain.getNumOfTotalInstanceInDomain(), targetDomain.getNumOfTrainInstanceInDomain(), targetDomain.getNumOfTestInstanceInDomain() ))    
    
    # Parsing train and test data for source domain
    trainXsource, trainYsource  = parseProcessedDataFileForScikit.parseDataFile(sourceDomain.getTrainFileFullPath())
    testXsource, testYsource = parseProcessedDataFileForScikit.parseDataFile(sourceDomain.getTestFileFullPath())
    trainSourceSize = len(trainXsource)
    vectorizer = DictVectorizer(dtype=float, sparse=True)
    encoder = LabelEncoder()
    
    # Parsing train and test data for target domain
    trainXtarget, trainYtarget  = parseProcessedDataFileForScikit.parseDataFile(targetDomain.getTrainFileFullPath())
    testXtarget, testYtarget = parseProcessedDataFileForScikit.parseDataFile(targetDomain.getTestFileFullPath())
    trainTargetSize = len(trainXtarget)
    
    # Vectorize!
    print("\nVectorizing train sets of source and target domains.")
    vectorized = vectorizer.fit_transform(trainXsource+trainXtarget)
    vectorizedLabels = encoder.fit_transform(trainYsource+trainYtarget)
    total = trainSourceSize+trainTargetSize
    numOfFeatures = vectorized[0].get_shape()[1]
    print("Vectorizer num of features: %d" % numOfFeatures)
    print("data set type is %s" % type(vectorized))
    print("data set instance is %s" % type(vectorized[3]))
    
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
    print("testY type is %s" % type(newTestYtarget))
    
    #Build feature sentiment dict
    '''
    print("Building feature sentiment dict")
    featureNames = vectorizer.get_feature_names()
    sent = SentimentWordFrequencyModel()
    featSentDict = {}
    
    for i in range(len(featureNames)):
        word = featureNames[i]
        if '_' not in word:
            featSentDict[i]  = sent.getSentimentOfWord(word)
    '''
    #print(featSentDict)
    
    '''
    #Check how to de-vectorize
    #print("Vectorizer.get_feature_names():")
    #print(vectorizer.get_feature_names())
    print("Check how to de-vectorize")
    sample = newTestXsource[0][0]
    print("Vectorized sample:")
    print(type(sample))
    print(sample)
    print("sample.shape[0] = %d" % sample.shape[0])
    print("sample.shape[1] = %d" % sample.shape[1])
    print("De-Vectorized sample:")
    for i in range(sample.shape[1]):
        if (sample[0,i] != 0):
            #print("%d, %f" % (i,sample[0,i]))
            word = vectorizer.get_feature_names()[i]
            print("%s, %f" % (word,sample[0,i]))
    '''
    
    # Package train and test sets
    newTrainSource = testActiveLearner.ActiveLearnerTester.dataType(newTrainXsource, newTrainYsource)
    newTestSource = testActiveLearner.ActiveLearnerTester.dataType(newTestXsource, newTestYsource)
    newTrainTarget = testActiveLearner.ActiveLearnerTester.dataType(newTrainXtarget, newTrainYtarget)
    newTestTarget = testActiveLearner.ActiveLearnerTester.dataType(newTestXtarget, newTestYtarget)
    
    # Package domains
    newSourceDomain = testActiveLearner.ActiveLearnerTester.domainType(sourceDomain.value, newTrainSource, newTestSource)
    newTargetDomain = testActiveLearner.ActiveLearnerTester.domainType(targetDomain.value, newTrainTarget, newTestTarget)
    
    classifiersToRun = ActiveLearnerTester.classifiersToRunType(False, False, False, False, False, True, False) #Runing only Sentiment Intensity
    bathConfig = ActiveLearnerTester.bathConfigType(10,[35,40])
    partialTrainConfig = None
    testActiveLearner.testActiveLearners(newSourceDomain, newTargetDomain, classifiersToRun = classifiersToRun, bathConfig = bathConfig, partialTrainConfig = partialTrainConfig, vectorizer = vectorizer)
    
def testSomeSpecificCombination(source, target):
    print("testing from %s to %s" % (source.value, target.value))
    testActiveLearnersWithBlitzerDomains(source, target)
    #testActiveLearnersWithBlitzerDomains(source, target, batchRange = range(5,15,5))

testSomeSpecificCombination(BlitzerDatasetDomain.outdoor, BlitzerDatasetDomain.sports)