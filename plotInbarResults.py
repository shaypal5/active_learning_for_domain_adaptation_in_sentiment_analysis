# -*- coding: utf-8 -*-

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

#General style for all plots
with plt.style.context('fivethirtyeight'):
        
    #For alpha = 1
    print("For alpha = 1")
    fig1 = plt.figure(1)
    fig1.set_size_inches(9,6)
    fig1.suptitle('Alpha = 1', fontsize=12)
    al1 = fig1.add_subplot(111)
    values = [200,400,600,800,1000,1200,1400,1600,1800,2000,2200,2400,2600]
    #values = [0.07692307692307693, 0.16666666666666666, 0.2727272727272727, 0.4, 0.5555555555555556, 0.75, 1.0, 1.3333333333333333, 1.8, 2.5, 3.6666666666666665, 6.0, 13.0]
    sourceAcc = [0.4825, 0.5108333333333334, 0.4725, 0.48333333333333334, 0.5091666666666667, 0.4975, 0.5133333333333333, 0.5991666666666666, 0.6108333333333333, 0.6283333333333333, 0.6625, 0.6058333333333333, 0.6275]
    targAcc = [0.7208333333333333, 0.6533333333333333, 0.6291666666666667, 0.6558333333333334, 0.6283333333333333, 0.5741666666666667, 0.5508333333333333, 0.49, 0.5008333333333334, 0.5041666666666667, 0.5158333333333334, 0.49, 0.4825]
    UncAcc = [0.6358333333333334, 0.6525, 0.6958333333333333, 0.6966666666666667, 0.7283333333333334, 0.6158333333333333, 0.6758333333333333, 0.6866666666666666, 0.7166666666666667, 0.7383333333333333, 0.655, 0.6291666666666667, 0.7083333333333334]
    source, = al1.plot(values, sourceAcc)
    target, = al1.plot(values, targAcc)
    unc, = al1.plot(values, UncAcc)
    plt.ylim(0.45,0.77)
    #uncAcuPlt.set_xlabel('Number of instances', fontsize=12, color='k')
    #uncAcuPlt.set_ylabel('Accuracy', fontsize=12, color='k')
    al1.tick_params(axis='x', labelcolor='k', labelsize = 18)#, colors='k',)
    al1.tick_params(axis='y', labelcolor='k', labelsize = 18)#, colors='k', size=20)

    #Legend    
    lg = fig1.legend( handles=[source, target, unc],labels = ['SourceSVM', 'TargetSVM', 'Uncertainty'], prop={'size':17} ) 
    lg.draw_frame(True)
    frame = lg.get_frame()
    frame.set_ec('0.45')
    frame.set_linewidth(1)
    plt.savefig('alpha_1.pdf')    
    plt.show()
    
print("finished plotting")

