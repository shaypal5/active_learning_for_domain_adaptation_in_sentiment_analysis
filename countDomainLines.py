# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 16:49:04 2014

@author: Shay
"""
from BlitzerDatasetDomain import BlitzerDatasetDomain

def countAllDomainLines():
    print("starting to count domain lines")
    for domain in BlitzerDatasetDomain:
        count = 0
        file = open(domain.getBalancedFileFullPath())
        for line in file:
            count += 1
        print(domain.value + ' has '+str(count)+' instances.')
        file.close()
    
def countDomainLines(domain):
    count = 0
    file = open(domain.getBalancedFileFullPath())
    for line in file:
        count += 1
    print(domain.value + ' has '+str(count)+' instances.')
    file.close()
    return count