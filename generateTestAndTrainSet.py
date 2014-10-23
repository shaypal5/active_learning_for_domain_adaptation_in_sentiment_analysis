# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 20:31:35 2014

@author: Shay
"""

import re
from BlitzerDatasetDomain import BlitzerDatasetDomain

labelRegex = '(#label#)(:)(negative|positive)'

def countPosInFile(filePath):
    file = open(filePath)
    
    total = 0
    posNum = 0
    for line in file:
        total += 1
        if re.search(labelRegex,line).group(3) == 'positive':
            posNum += 1
    file.close()
    return total,posNum

def generateTestAndTrainSet(domain):
    print("Generating test and train set for domain %s" % domain.value)
    TRAIN_PERCENT = 0.7
    trainSet = []
    testSet = []
    total, posNum = countPosInFile(domain.getBalancedFileFullPath())
    negNum = total - posNum
    print("Data file contains a total of "+str(total)+" instances.")
    print("Of those "+str(posNum)+" are labeled Positive and "+str(negNum)+" are labeled Negative.")
    
    posCount = 0
    negCount = 0    
    file = open(domain.getBalancedFileFullPath())
    
    for line in file:
        #print("here 0")
        if re.search(labelRegex,line).group(3) == 'positive':
            #print("here1")
            if posCount < round(posNum * TRAIN_PERCENT):
                trainSet.append(line)
            else:
                testSet.append(line)
            posCount += 1
        else:
            if negCount < round(negNum * TRAIN_PERCENT):
                trainSet.append(line)
            else:
                testSet.append(line)
            negCount += 1
    file.close()
    print("Finished building train and test sets")
    print("train size = "+str(len(trainSet)))
    print("test size = "+str(len(testSet)))
            
        #balancedDataSet.append(line)
    
    #print("posNum = "+str(posNum))

    #write train set file    
    print("Writing train set to file")
    trainFile = open(domain.getTrainFileFullPath(),'w+')
    for line in trainSet:
            trainFile.write(line)
    trainFile.flush()
    trainFile.close()
    
     #write test set file
    print("Writing test set to file")
    testFile = open(domain.getTestFileFullPath(),'w+')
    for line in testSet:
            testFile.write(line)
    testFile.flush()
    testFile.close()

#for domain in BlitzerDatasetDomain:
#    generateTestAndTrainSet(domain)