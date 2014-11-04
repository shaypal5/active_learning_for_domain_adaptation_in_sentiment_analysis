# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

#Uncertainty selectr accuracy plot
print("Uncertainty selector accuracy plot")
with plt.style.context('fivethirtyeight'):
    fig = plt.figure(1)
    #fig.suptitle('Accuracy of Uncertainty Selection', fontsize=14)
    uncAcuPlt = fig.add_subplot(111)
    values = [0,20,50,100,150,200,250,300,350,400,450,500]
    uncAcuPlt.plot(values, [0.7307 for i in range(12)], 'r--')
    uncAcuPlt.plot(values, [0.8876 for i in range(12)], 'b--')
    UncAcu = [None,0.7388,0.7807,0.7807,0.7997,0.8024,0.824,0.8234,0.8376,0.8477,0.8518,0.8484]
    uncAcuPlt.plot(values, UncAcu)
    plt.ylim(0.5,1)
    uncAcuPlt.set_xlabel('Number of instances', color='k', fontsize=12)
    uncAcuPlt.set_ylabel('Accuracy', color='k', fontsize=12)
    plt.show()


    #General accruacy bar graph
    N = 8 #the number of groups
    ind = np.arange(N)  # the x locations for the groups
    width = 0.15       # the width of the bars
    fig, ax = plt.subplots()
    fig.set_size_inches(9,5)
    
    sourceAcc = (0.9674, 1.084, 1.02298850574713, 0.978082824, 0.879955539, 0.892857143, 0.91954023, 0.831460674)
    rects1 = ax.bar(ind, sourceAcc, width, color='#E24A33' )
    uncertainAcc = (0.97399834534925, 1.08433734939759, 1.02298850574713, 1.006575153, 1.004940101, 1, 1.011494253, 0.95505618)
    rects2 = ax.bar(ind+width, uncertainAcc, width, color='#7A68A6')
    partialAcc = (0.9935, 1.08433734939759, 0.988888888888889, 1, 0.984932691, 1.011904762, 0.988505747, 0.943820225)
    rects3 = ax.bar(ind+2*width, partialAcc, width, color='#188487')
    stqbcAcc = (0.9935, 1.07228915662651, 1.03448275862069, 0.995616565, 0.924910461, 0.952380952, 1.011494253, 0.95505618)
    rects4 = ax.bar(ind+3*width, stqbcAcc, width)#, color='#467821')
    sentIntAcc = (0.973998345, 1.089638554, 1.028850575, 0.989041412, 0.97999259, 0.971785714, 0.973793103, 0.928202247)
    rects5 = ax.bar(ind+4*width, sentIntAcc, width, color='y')
    
    # add some text for labels, title and axes ticks
    ax.set_ylabel('Relative Accuracy', color='k', fontsize=12)
    #ax.set_title('Scores by group and gender', color='k')
    ax.set_xticks(ind+2*width)
    ax.set_xticklabels( ('Similar \n Big To Small', 'Different', 'Similar \n Big To Big', 'Different', 'Similar \n Small To Small', 'Different', 'Similar \n Small To Big', 'Different') , color='k', fontsize=12)
    
    lg = ax.legend( (rects1[0], rects2[0], rects3[0], rects4[0], rects5[0]), ('SourceSVM', 'Uncertainty', 'PartialQBC', 'STQBC', 'SentimentIntensity') )
    lg.draw_frame(True)
    frame = lg.get_frame()
    frame.set_ec('0.45')
    frame.set_linewidth(1)
    ax.axhline(y=1, linewidth=2, color='k', zorder=0, ls='dashed')
    
    #autolabel(rects1)
    #autolabel(rects2)
    #autolabel(rects2)
    #autolabel(rects2)
    plt.ylim(0.8,1.1)
    plt.show()

