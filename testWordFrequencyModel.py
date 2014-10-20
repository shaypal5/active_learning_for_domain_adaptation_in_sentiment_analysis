# -*- coding: utf-8 -*-
"""
Created on Sat Oct 18 15:00:23 2014

@author: inbar
"""

from sklearn.svm import LinearSVC
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder

from BlitzerDatasetDomain import BlitzerDatasetDomain
from WordFrequencyModel import WordFrequencyModel
import parseProcessedDataFileForScikit

domain = BlitzerDatasetDomain.books

print('Testing the WordFrequencyModel class')
trainX, trainY = parseProcessedDataFileForScikit.parseDataFile(domain.getTrainFileFullPath())
print('finished parsing data')

babyWordFreqModel = WordFrequencyModel()
babyWordFreqModel.processDomain(trainX, trainY)
babyWordFreqModel.printModelDetails()

print("Generating new instance:")
newInst = babyWordFreqModel.generateInstance()
print("Instance length: %d" % len(newInst[0]))
#print(newInst[0])

#======================================================
#   comparing with non-generated data taught SVM
#======================================================

#Get non-generated train and test sets
#already got trainX and trainY above
#trainX, trainY = parseProcessedDataFileForScikit.parseDataFile(domain.getTrainFileFullPath())
testX, testY = parseProcessedDataFileForScikit.parseDataFile(domain.getTestFileFullPath())
trainSize = len(trainX)

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
generatedX, generatedY = babyWordFreqModel.generateDataset(trainSize)
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