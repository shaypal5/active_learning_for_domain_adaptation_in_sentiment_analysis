# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 17:12:13 2014

@author: inbar
"""

def read_mat( file_path ):
    import numpy as np
    mat = open(file_path, 'r')
    mat.next() # % Size = 30 30
    length = int(mat.next().split()[-1])
    mat.next() # zzz = zeros(18,3)
    mat.next() # zzz = [
    ans = np.array([ map(float, mat.next().split()) for i in xrange(length) ])
    mat.close()
    return ans

if __name__ == '__main__':
    import scipy.io
    f = 'C:\\Users\\inbar\\Dropbox\\1 - MSc\\Modern Statistical Data Analysis\\Final Project\\W.mat'
  #  mat = read_mat(f);
    mat = scipy.io.loadmat(f)
    print(mat)