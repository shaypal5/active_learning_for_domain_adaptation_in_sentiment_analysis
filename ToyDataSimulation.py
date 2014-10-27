# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 21:34:37 2014

@author: inbar
"""

import simulateGaussianDomains as dataSimulator
import xalglib
import numpy as np

def main():
    n = 10
    numOfSamples = 10
    
    #generate P(X|Y=1) and P(X|Y=0) for source domain
    dist = dataSimulator.generateSourceDistributions(n)
    sourceP0 = dist.P0; sourceP1 = dist.P1
    
    #check that the distribution are different enough
 #   bhCoeff = dataSimulator.getBhattacharyyaCoefficient(sourceP0, sourceP1)
 #   print(bhCoeff)
    
    #generate P(X|Y=1) and P(X|Y=0) for target domain
    dist = dataSimulator.generateTargetDistributions(sourceP0, sourceP1)
    targetP0 = dist.P0; targetP1 = dist.P1
        
    posTargetSamples = targetP1.getSamples(numOfSamples)
    negTargetSamples = targetP0.getSamples(numOfSamples) 
    posSourceSamples = sourceP1.getSamples(numOfSamples)
    negSourceSamples = sourceP0.getSamples(numOfSamples)

    
if __name__ == '__main__':
    main()