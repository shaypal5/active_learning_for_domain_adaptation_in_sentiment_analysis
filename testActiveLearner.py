# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 18:33:19 2014

@author: Shay
"""

from UncertaintySampleSelector import UncertaintySampleSelector
from QueryByPartialDataCommiteeSampleSelector import QueryByPartialDataCommiteeSampleSelector
from TargetAndSourceQBCSampleSelector import TargetAndSourceQBCSampleSelector
from BlitzerDatasetDomain import BlitzerDatasetDomain

import ActiveLearner
import parseProcessedDataFileForScikit

from sklearn.svm import LinearSVC
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder

sourceDomain = BlitzerDatasetDomain.apparel
targetDomain = BlitzerDatasetDomain.jewelry

#Parsing train and test data for source domain
trainXsource, trainYsource  = parseProcessedDataFileForScikit.parseDataFile(sourceDomain.getTrainFileFullPath())
testXsource, testYsource = parseProcessedDataFileForScikit.parseDataFile(sourceDomain.getTestFileFullPath())
trainSourceSize = len(trainXsource)
vectorizer = DictVectorizer(dtype=float, sparse=True)
encoder = LabelEncoder()

#Parsing train and test data for target domain
trainXtarget, trainYtarget  = parseProcessedDataFileForScikit.parseDataFile(targetDomain.getTrainFileFullPath())
testXtarget, testYtarget = parseProcessedDataFileForScikit.parseDataFile(targetDomain.getTestFileFullPath())
trainTargetSize = len(trainXtarget)
print(type(trainXtarget))

#vectorize!
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
#selector = QueryByPartialDataCommiteeSampleSelector(sourceClassifier)
selector = TargetAndSourceQBCSampleSelector(sourceClassifier)

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