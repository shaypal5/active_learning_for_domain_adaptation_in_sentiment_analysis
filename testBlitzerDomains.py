# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 11:14:31 2014

@author: Shay
"""

import parseProcessedDataFileForScikit

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder

def testActiveLearnersWithBlitzerDomains(sourceDomain, targetDomain):
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
    
    # Package train and test sets
    newTrainSource = (newTrainXsource, newTrainYsource)
    newTestSource = (newTestXsource, newTestYsource)
    newTrainTarget = (newTrainXtarget, newTrainYtarget)
    newTestTarget = (newTestXtarget, newTestYtarget)
    
    # Package domains
    newSourceDomain = (sourceDomain.value, newTrainSource, newTestSource)
    newTargetDomain = (targetDomain.value, newTrainTarget, newTestTarget)