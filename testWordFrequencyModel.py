# -*- coding: utf-8 -*-
"""
Created on Sat Oct 18 15:00:23 2014

@author: inbar
"""

from BlitzerDatasetDomain import BlitzerDatasetDomain
from WordFrequencyModel import WordFrequencyModel
import parseProcessedDataFileForScikit

print('Testing the WordFrequencyModel class')
Xbaby, Ybaby = parseProcessedDataFileForScikit.parseDataFile(BlitzerDatasetDomain.baby.getBalancedFileFullPath())
print('finished parsing data')

babyWordFreqModel = WordFrequencyModel()
babyWordFreqModel.processDomain(Xbaby, Ybaby)
babyWordFreqModel.printModelDetails()

print("Generating new instance:")
newInst = babyWordFreqModel.generateInstance()
print("Instance length: %d" % len(newInst[0]))
print(newInst[0])

size = 100
print("Generating data set of size %d" % size)
babyWordFreqModel.generateDataset(size)
print("Done generating dataset")
