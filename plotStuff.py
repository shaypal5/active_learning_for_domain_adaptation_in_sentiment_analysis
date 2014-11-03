# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

#Uncertainty selectr accuracy plot
print("Uncertainty selector accuracy plot")
fig = plt.figure(1)
#fig.suptitle('Accuracy of Uncertainty Selection', fontsize=14)
uncAcuPlt = fig.add_subplot(111)
values = [0,20,50,100,150,200,250,300,350,400,450,500]
uncAcuPlt.plot(values, [0.7307 for i in range(12)], 'r--')
uncAcuPlt.plot(values, [0.8876 for i in range(12)], 'b--')
UncAcu = [None,0.7388,0.7807,0.7807,0.7997,0.8024,0.824,0.8234,0.8376,0.8477,0.8518,0.8484]
uncAcuPlt.plot(values, UncAcu)
plt.ylim(0.5,1)
uncAcuPlt.set_xlabel('Number of instances')
uncAcuPlt.set_ylabel('Accuracy')
plt.show()