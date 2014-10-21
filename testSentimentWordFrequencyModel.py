# -*- coding: utf-8 -*-
"""
Created on Sat Oct 18 15:00:23 2014

@author: inbar
"""

from sklearn.svm import LinearSVC
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder

from BlitzerDatasetDomain import BlitzerDatasetDomain
from SentimentWordFrequencyModel import SentimentWordFrequencyModel
import parseProcessedDataFileForScikit

#======================================================
#   comparing with non-generated data taught SVM
#======================================================
domain = BlitzerDatasetDomain.apparel

print('Testing the SentimentWordFrequencyModel class')
#Get non-generated train and test sets
trainX, trainY = parseProcessedDataFileForScikit.parseDataFile(domain.getTrainFileFullPath())
testX, testY = parseProcessedDataFileForScikit.parseDataFile(domain.getTestFileFullPath())
trainSize = len(trainX)
print('finished parsing data')

domainWordFreqModel = SentimentWordFrequencyModel()
domainWordFreqModel.processDomain(trainX, trainY)
#domainWordFreqModel.printModelDetails()
print('finished building model')

vectorizer = DictVectorizer(dtype=float, sparse=True)
encoder = LabelEncoder()
vectorizedTrainX = vectorizer.fit_transform(trainX)
vectorizedTrainY = encoder.fit_transform(trainY)

print("Test real-data-trained classifier")
classifier = LinearSVC()
classifier.fit(vectorizedTrainX,vectorizedTrainY)
correct = 0
wrong = 0
vectorizedTestX = vectorizer.transform(testX)
classes = encoder.classes_
for i in range(len(testX)):
    prediction = classifier.predict(vectorizedTestX[i])
    if classes[prediction] == testY[i]:
        correct += 1
    else:
        wrong += 1
print("Got "+str(correct)+" correct.")
print("Got "+str(wrong)+" wrong.")

#Now checking an SVM with generated data
print("Generating data set of size %d" % trainSize)
generatedX, generatedY = domainWordFreqModel.generateDataset(trainSize)
print("generated %d instances" % len(generatedX))
print("generated %d labels" % len(generatedY))
#print(generatedY)
print("Done generating dataset")

vectorizer = DictVectorizer(dtype=float, sparse=True)
encoder = LabelEncoder()
vectorizedGeneratedX = vectorizer.fit_transform(generatedX)
vectorizedGeneratedY = encoder.fit_transform(generatedY)

print("Test generated-data-trained classifier")
classifier = LinearSVC()
classifier.fit(vectorizedGeneratedX,vectorizedGeneratedY)
correct = 0
wrong = 0
vectorizedTestX = vectorizer.transform(testX)
classes = encoder.classes_
for i in range(len(testX)):
    prediction = classifier.predict(vectorizedTestX[i])
    if classes[prediction] == testY[i]:
        correct += 1
    else:
        wrong += 1
print("Got "+str(correct)+" correct.")
print("Got "+str(wrong)+" wrong.")