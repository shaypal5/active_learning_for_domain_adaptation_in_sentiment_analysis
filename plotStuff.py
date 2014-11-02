# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

#Uncertainty selectr accuracy plot
uncAcuPlt = plt.figure(1)
uncAcuPlt.suptitle('Accuracy of Uncertainty Selection', fontsize=14)
values = [0,20,50,100,150,200,250,300,350,400,450,500]
plt.plot(values, [0.7307 for i in range(12)], 'r--')
plt.plot(values, [0.8876 for i in range(12)], 'b--')
UncAcu = [None,0.7388,0.7807,0.7807,0.7997,0.8024,0.824,0.8234,0.8376,0.8477,0.8518,0.8484]
plt.plot(values, UncAcu)
plt.ylim(0.5,1)