# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 17:28:08 2014

@author: inbar
"""
import scipy.io
import scipy
import numpy as np
import xalglib
import math
import collections

class multivariateGaussian:
    
    def __init__(self, mu, covarianceMatrix):
        self.mu = mu;
        self.covarianceMatrix = covarianceMatrix
    
    def getMu(self):
        return self.mu
    
    def setMu(self, mu):
        self.mu = mu
    
    def getCovariance(self):
        return self.covarianceMatrix
    
    def setCovariance(self, covarianceMatrix):
        self.covarianceMatrix = covarianceMatrix
    
    def getSamples(self, numOfSamples = 1):
        return np.random.multivariate_normal(self.mu, self.covarianceMatrix, numOfSamples)


def generateSourceDistributions(dimension):
    '''
    generate multivariate gaussian distributions, for specific dimension
    '''
    MATRIX_VARIABLE_NAME = 'W'
    covarianceMatrixFile = 'uHellingerGMMs\\W.mat'
    meansDiff = 4
    
    #set parameters for Y=0 distribution
    covMat0 = np.identity(dimension)
    mu0 = np.zeros(dimension)

    #set parmeters for Y=1 distribution
    covMat1 = scipy.io.loadmat(covarianceMatrixFile)
    covMat1 = covMat1[MATRIX_VARIABLE_NAME]
    mu1 = np.ones(dimension) * meansDiff
    
    P0 = multivariateGaussian(mu0, covMat0)
    P1 = multivariateGaussian(mu1, covMat1)
    
    distributions = collections.namedtuple('Distributions', ['P0', 'P1'])
    return distributions(P0, P1)
    
    
def generateTargetDistributions(sourceP0, sourceP1):
    '''
    generate target distribution according to source distribution
    '''
    dimension = len(sourceP0.getMu())
    rotationMatrix = xalglib.rmatrixrndorthogonal(dimension)
    
    targetP0 = multivariateGaussian( 
        np.dot(sourceP0.getMu(), rotationMatrix),  
        np.dot(np.dot(rotationMatrix, sourceP0.getCovariance()) , np.transpose(rotationMatrix)))        
    targetP1 = multivariateGaussian( 
        np.dot(sourceP1.getMu(), rotationMatrix),  
        np.dot(np.dot(rotationMatrix, sourceP1.getCovariance()) , np.transpose(rotationMatrix)))
    
    distributions = collections.namedtuple('Distributions', ['P0', 'P1'])
    return distributions(targetP0, targetP1)
 
   
def getBhattacharyyaCoefficient(P0, P1):
    bhDist = gau_bh(P0.getMu(), P0.getCovariance(), P1.getMu(), P1.getCovariance())
    bhCoeff = math.pow(math.e, -bhDist)
    return bhCoeff


def gau_bh(pm, pv, qm, qv):
    """
    Classification-based Bhattacharyya distance between two Gaussians
    with diagonal covariance.  Also computes Bhattacharyya distance
    between a single Gaussian pm,pv and a set of Gaussians qm,qv.
    pm = p mean ; pv = p variance
    """
    if (len(qm.shape) == 2):
        axis = 1
    else:
        axis = 0
    # Difference between means pm, qm
    diff = qm - pm
    # Interpolated variances
    pqv = (pv + qv) / 2.
    # Log-determinants of pv, qv
    ldpv = np.log(pv).sum()
    ldqv = np.log(qv).sum(axis)
    # Log-determinant of pqv
    ldpqv = np.log(pqv).sum(axis)
    # "Shape" component (based on covariances only)
    # 0.5 log(|\Sigma_{pq}| / sqrt(\Sigma_p * \Sigma_q)
    norm = 0.5 * (ldpqv - 0.5 * (ldpv + ldqv))
    # "Divergence" component (actually just scaled Mahalanobis distance)
    # 0.125 (\mu_q - \mu_p)^T \Sigma_{pq}^{-1} (\mu_q - \mu_p)
    dist = 0.125 * (diff * (1./pqv) * diff).sum(axis)
    return dist + norm

if __name__ == '__main__':
    res = generateSourceDistribution()
    print(res)