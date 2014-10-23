# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 16:09:29 2014

@author: Shay
"""
import countDomainLines
import generateTestAndTrainSet
import re

def generateSmallerDomainBalancedFile(domain, newSize):
    totalNum = countDomainLines.countDomainLines(domain)
    if newSize >= totalNum:
        return "Requested size is bigger than original balanced file size!"
    
    #total, posNum = generateTestAndTrainSet.countPosInFile(domain.getBalancedFileFullPath())
    newPosNum = int(round(newSize/2))
    
    posCount = 0
    negCount = 0  
    newBalanced = []  
    file = open(domain.getBalancedFileFullPath())
    
    for line in file:
        #print("here 0")
        if re.search(generateTestAndTrainSet.labelRegex, line).group(3) == 'positive':
            #print("here1")
            if posCount < newPosNum:
                newBalanced.append(line)
                posCount += 1
        else:
            if negCount < newPosNum:
                newBalanced.append(line)
                negCount += 1
    file.close()
    print("Finished building new, smaller, balanced file, sized %d, for domain %s" % (newSize, domain.value))
    
    #write train set file    
    print("Writing new balanced file set to file")
    newFile = open(domain.getBalancedFileFullPath()+'.'+str(newSize),'w+')
    for line in newBalanced:
            newFile.write(line)
    newFile.flush()
    newFile.close()