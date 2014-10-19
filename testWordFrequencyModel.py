# -*- coding: utf-8 -*-
"""
Created on Sat Oct 18 15:00:23 2014

@author: inbar
"""

from BlitzerDatasetDomain import BlitzerDatasetDomain
from WordFrequencyModel import WordFrequencyModel
import parseProcessedDataFileForScikit

Xbaby, Ybaby = parseProcessedDataFileForScikit.parseDataFile(BlitzerDatasetDomain.baby.getBalancedFileFullPath())
print('finished parsing data')

babyWordFreqModel = WordFrequencyModel()
babyWordFreqModel.processDomain(Xbaby, Ybaby)
babyWordFreqModel.printModelFrequencies()