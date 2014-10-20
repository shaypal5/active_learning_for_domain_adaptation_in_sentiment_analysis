# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 13:42:14 2014

@author: Shay
"""
import re

def parseDataFile(dataFile):
    file = open(dataFile)
    data = []
    for line in file:
        data.append(parseLine(line))
    return data
    
def parseLine(line):
    tokenCountRegex = '([^#:\s]+)(:)([1-9]+)'
    labelRegex = '(#label#)(:)(negative|positive)'
    if re.search(labelRegex,line).group(3) == 'positive':
        label = 1
    else:
        label = 0
    #print("label = "+str(label))
    pattern = re.compile(tokenCountRegex)
    return [dict([(token, True) for (token, dots, foroccurences) in re.findall(pattern, line)]),label]
        