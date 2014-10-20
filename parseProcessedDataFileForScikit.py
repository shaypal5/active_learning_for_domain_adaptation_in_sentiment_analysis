# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 13:42:14 2014

@author: Shay
"""
import re

def parseDataFile(dataFile):
    file = open(dataFile)
    X = []
    y = []
    for line in file:
        featureSet, label = parseLine(line)
        X.append(featureSet)
        y.append(label)    
    file.close()
    return [X,y]
    
def parseLine(line):
    tokenCountRegex = '([^#:\s]+)(:)([1-9]+)'
    labelRegex = '(#label#)(:)(negative|positive)'
    if re.search(labelRegex,line).group(3) == 'positive':
        label = 1
    else:
        label = 0
    #print("label = "+str(label))
    pattern = re.compile(tokenCountRegex)
    return [dict([(token, float(occurences)) for (token, dots, occurences) in re.findall(pattern, line)]),label]
    #return [dict([(token, True) for (token, dots, occurences) in re.findall(pattern, line)]),label]
        