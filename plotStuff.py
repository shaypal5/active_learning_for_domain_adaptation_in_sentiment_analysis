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

with plt.style.context('fivethirtyeight'):
    #General accruacy bar graph
    N = 8 #the number of groups
    ind = np.arange(N)  # the x locations for the groups
    width = 0.2       # the width of the bars
    fig, ax = plt.subplots()
    
    sourceAcc = (0.9674, 1.084, 1.012048193, 0.978082824, 0.879955539, 0.892857143, 0.91954023, 0.831460674)
    rects1 = ax.bar(ind, sourceAcc, width, color='#E24A33' )
    uncertainAcc = (0.9739, 1.084, 1.012048193, 1.006575153, 1.004940101, 1, 1.011494253, 0.95505618)
    rects2 = ax.bar(ind+width, uncertainAcc, width, color='#7A68A6')
    partialAcc = (0.9934, 1.084, 1.024096386, 1, 0.984932691, 1.011904762, 0.988505747, 0.943820225)
    rects3 = ax.bar(ind+2*width, partialAcc, width, color='#188487')
    stqbcAcc = (0.9934, 1.072, 1.012048193, 0.995616565, 0.924910461, 0.952380952, 1.011494253, 0.95505618)
    rects4 = ax.bar(ind+3*width, stqbcAcc, width)#, color='#467821')
    
    # add some text for labels, title and axes ticks
    ax.set_ylabel('Scores')
    ax.set_title('Scores by group and gender')
    ax.set_xticks(ind+3*width)
    ax.set_xticklabels( ('G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8') )
    
    ax.legend( (rects1[0], rects2[0], rects3[0], rects4[0]), ('SourceSVM', 'Uncertainty', 'partialQBC', 'STQBC') )
    
    #autolabel(rects1)
    #autolabel(rects2)
    #autolabel(rects2)
    #autolabel(rects2)
    plt.ylim(0.5,1.1)
    plt.show()

