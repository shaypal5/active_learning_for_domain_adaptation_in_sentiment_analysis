# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 18:33:19 2014

@author: Shay
"""

from UncertaintySampleSelector import UncertaintySampleSelector
from QueryByPartialDataCommiteeSampleSelector import QueryByPartialDataCommiteeSampleSelector
import ActiveLearner
import parseProcessedDataFileForScikit

from sklearn.svm import LinearSVC
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder

sourceFolderPath = "C:/OrZuk/Data/sorted_data/apparel/"
targetFolderPath = "C:/OrZuk/Data/sorted_data/jewelry_&_watches/"
trainFileName = "processed.review.trainset"
testFileName = "processed.review.testset"

#Parsing train and test data for domain 1 - Apparel
trainXsource, trainYsource  = parseProcessedDataFileForScikit.parseDataFile(sourceFolderPath+trainFileName)
testXsource, testYsource = parseProcessedDataFileForScikit.parseDataFile(sourceFolderPath+testFileName)
trainSourceSize = len(trainXsource)
vectorizer = DictVectorizer(dtype=float, sparse=True)
encoder = LabelEncoder()

#vectorize!
#newTrainXapparel = vectorizer.fit_transform(trainXapparel)
#newTrainYapparel = encoder.fit_transform(trainYapparel)

#Parsing train and test data for domain 2 - Jewlery
trainXtarget, trainYtarget  = parseProcessedDataFileForScikit.parseDataFile(targetFolderPath+trainFileName)
testXtarget, testYtarget = parseProcessedDataFileForScikit.parseDataFile(targetFolderPath+testFileName)
trainTargetSize = len(trainXtarget)
print(type(trainXtarget))

#vectorize!
#newTrainXjewlery = vectorizer.transform(trainXjewlery)
#newTrainYjewlery = encoder.transform(trainYjewlery)
vectorized = vectorizer.fit_transform(trainXsource+trainXtarget)
vectorizedLabels = encoder.fit_transform(trainYsource+trainYtarget)
total = trainSourceSize+trainTargetSize

print(type(vectorized))
print(type(vectorized[0]))
print("num of features: "+str(vectorized[0].get_shape()[1]))
numOfFeatures = vectorized[0].get_shape()[1]

for i in range(total):
    if vectorized[i].get_shape()[1] != numOfFeatures:
        print("found different: "+str(vectorized[i].get_shape()[1]))

newTrainXsource = vectorized[0:trainSourceSize]
newTrainYsource = vectorizedLabels[0:trainSourceSize]
newTrainXtarget = vectorized[trainSourceSize+1:total]
newTrainYtarget = vectorizedLabels[trainSourceSize+1:total]
#print(newTrainXapparel)

#train classifier on source domain
sourceClassifier = LinearSVC()
sourceClassifier.fit(newTrainXsource,newTrainYsource)

#train classifier on target domain
targetClassifier = LinearSVC()
targetClassifier.fit(newTrainXtarget,newTrainYtarget)

#selector = UncertaintySampleSelector()
selector = QueryByPartialDataCommiteeSampleSelector(sourceClassifier)

learner = ActiveLearner.ActiveLearner(selector)
resultClassifier = learner.train(sourceClassifier,[newTrainXsource,newTrainYsource],[newTrainXtarget,newTrainYtarget])

print("target classifier was trained on {0} labeled instances".format(len(newTrainYtarget)))

#Test source classifier
print("Test source classifier")
correct = 0
wrong = 0
newTestX = vectorizer.transform(testXtarget)
classes = encoder.classes_
for i in range(len(testXtarget)):
    prediction = sourceClassifier.predict(newTestX[i])
    if classes[prediction] == testYtarget[i]:
        correct += 1
    else:
        wrong += 1
print("Got "+str(correct)+" correct.")
print("Got "+str(wrong)+" wrong.")

#Test target classifier
print("Test target classifier")
correct = 0
wrong = 0
newTestX = vectorizer.transform(testXtarget)
classes = encoder.classes_
for i in range(len(testXtarget)):
    prediction = targetClassifier.predict(newTestX[i])
    if classes[prediction] == testYtarget[i]:
        correct += 1
    else:
        wrong += 1
print("Got "+str(correct)+" correct.")
print("Got "+str(wrong)+" wrong.")
        
#Test result classifier
print("Test result classifier")
correct = 0
wrong = 0
newTestX = vectorizer.transform(testXtarget)
classes = encoder.classes_
for i in range(len(testXtarget)):
    prediction = resultClassifier.predict(newTestX[i])
    if classes[prediction] == testYtarget[i]:
        correct += 1
    else:
        wrong += 1
print("Got "+str(correct)+" correct.")
print("Got "+str(wrong)+" wrong.")

print("Test done")