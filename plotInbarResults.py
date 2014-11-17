# -*- coding: utf-8 -*-

import socket
import matplotlib.pyplot as plt
from AlphaRes import AlphaRes
    
def getFolderPath():
    if (socket.gethostname() == 'InbarPC'):
        return('C:/Dropbox/University/Courses/Modern Statistical Data Anaysis/Final Project/Final Paper/')
    if (socket.gethostname() == 'Shay-Lenovo-Lap'):
        return('C:/Dropbox/University/Courses/Modern Statistical Data Anaysis/Final Project/Final Paper/')
    if (socket.gethostname() == 'ShayPalachy-PC'):
        return('C:/Dropbox/University/Courses/Modern Statistical Data Anaysis/Final Project/Final Paper/')
    if (socket.gethostname() == 'SNEEZY'):
        return('C:/Dropbox/University/Courses/Modern Statistical Data Anaysis/Final Project/Final Paper/')
    if (socket.gethostname() == 'Reem-PC'):
        return('C:/Dropbox/University/Courses/Modern Statistical Data Anaysis/Final Project/Final Paper/')
    return None

colorsArray = ['#00aedb', '#4cc88a', '#00B159', '#007b3e', '#de587a', '#d11141', '#920b2d'] # Metro UI-like pastels. blue for source, greens for general, reds for sentiment
reverse = True

#General style for all plots
with plt.style.context('fivethirtyeight'):
        
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4, 2, sharex='col', sharey='row')
    
    fig.set_size_inches(20,24.5)
    
    
    axi = [ax1, ax2, ax3, ax4, ax5, ax6 ,ax7, ax8]
    
    i = 0
    for alpha in AlphaRes.alphaRange:
        print("For alpha = %0.1f" % alpha)
        ax = axi[i]
        #fig1 = plt.figure(i)
        #fig1.set_size_inches(9,6)
        #fig1.suptitle('Alpha = '+str(alpha), fontsize=12)
        ax.set_title('Alpha = %0.1f, KL = %0.3f' % (alpha, AlphaRes.KL[i]), fontsize=24)
        #al1 = fig1.add_subplot(111)
        if reverse:
            AlphaRes.sourceAcc[i].reverse()
            AlphaRes.targAcc[i].reverse()
            AlphaRes.uncAcc[i].reverse()
        source, = ax.plot(AlphaRes.sourceTrainSizeValues, AlphaRes.sourceAcc[i], color = colorsArray[5])
        target, = ax.plot(AlphaRes.sourceTrainSizeValues, AlphaRes.targAcc[i], color = colorsArray[2])
        unc, = ax.plot(AlphaRes.sourceTrainSizeValues, AlphaRes.uncAcc[i], color = colorsArray[0])
        plt.ylim(0.4,0.8)
        #uncAcuPlt.set_xlabel('Number of instances', fontsize=12, color='k')
        #uncAcuPlt.set_ylabel('Accuracy', fontsize=12, color='k')
        ax.tick_params(axis='x', labelcolor='k', labelsize = 18)#, colors='k',)
        ax.tick_params(axis='y', labelcolor='k', labelsize = 18)#, colors='k', size=20)
        i += 1
        
    
    ax8.tick_params(axis='x', labelcolor='k', labelsize = 18)#, colors='k',)
    ax8.tick_params(axis='y', labelcolor='k', labelsize = 18)#, colors='k', size=20)
    
    #Legend    
    lg = fig.legend( handles=[source, target, unc],labels = ['SourceSVM', 'TargetSVM', 'Uncertainty'], prop={'size':29}, loc = (0.75,0.035) ) 
    lg.draw_frame(True)
    frame = lg.get_frame()
    frame.set_ec('0.45')
    frame.set_linewidth(1)
    plt.tight_layout()
    #alphaStr = str(alpha)
    #alphaStr = alphaStr.replace('.','')
    #plt.savefig('alpha_' + alphaStr + '.pdf')
    plt.savefig(getFolderPath()+'alpha_sim.pdf')
    plt.show()
    
print("finished plotting")

