# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from AlphaRes import AlphaRes

#General style for all plots
with plt.style.context('fivethirtyeight'):
    
    i = 0
    for alpha in AlphaRes.alphaRange:
        print("For alpha = %0.1f" % alpha)
        fig1 = plt.figure(i)
        fig1.set_size_inches(9,6)
        fig1.suptitle('Alpha = '+str(alpha), fontsize=12)
        al1 = fig1.add_subplot(111)
        source, = al1.plot(AlphaRes.sourceTrainSizeValues, AlphaRes.sourceAcc[i])
        target, = al1.plot(AlphaRes.sourceTrainSizeValues, AlphaRes.targAcc[i])
        unc, = al1.plot(AlphaRes.sourceTrainSizeValues, AlphaRes.uncAcc[i])
        plt.ylim(0.4,0.8)
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
        alphaStr = str(alpha)
        alphaStr = alphaStr.replace('.','')
        plt.savefig('alpha_' + alphaStr + '.pdf')    
        plt.show()
        i += 1
    
print("finished plotting")

