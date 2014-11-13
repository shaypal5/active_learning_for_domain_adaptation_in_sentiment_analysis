# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import socket
    
def getFolderPath():
    if (socket.gethostname() == 'InbarPC'):
        return('C:/Dropbox/University/Courses/Modern Statistical Data Anaysis/Final Project/Final Paper/')
    if (socket.gethostname() == 'Shay-Lenovo-Lap'):
        return('C:/Dropbox/University/Courses/Modern Statistical Data Anaysis/Final Project/Final Paper/')
    if (socket.gethostname() == 'ShayPalachy-PC'):
        return('C:/Dropbox/University/Courses/Modern Statistical Data Anaysis/Final Project/Final Paper/')
    if (socket.gethostname() == 'SNEEZY'):
        return('C:/OrZuk/Data/sorted_data/')
    if (socket.gethostname() == 'Reem-PC'):
        return('C:/OrZuk/Data/sorted_data/')
    return None

def plotAccuracyBarGraph(colorsArray):
    #General style for all plots
    with plt.style.context('fivethirtyeight'):
        
        #General accruacy bar graph
        N = 8 #the number of groups
        ind = np.arange(N)  # the x locations for the groups
        width = 0.12       # the width of the bars
        fig4, ax = plt.subplots()
        fig4.set_size_inches(9,5)
        fig4.suptitle('Big to Small    Big to Big    Small to Small    Small to Big', y=0.03, fontsize=15)
        
        sourceAcc = (0.9674, 1.084, 1.02298850574713, 0.978082824, 0.879955539, 0.892857143, 0.91954023, 0.831460674)
        rects1 = ax.bar(ind, sourceAcc, width, color=colorsArray[0])# '#e45c47' )
        uncertainAcc = (0.97399834534925, 1.08433734939759, 1.02298850574713, 1.006575153, 1.004940101, 1, 1.011494253, 0.95505618)
        rects2 = ax.bar(ind+width, uncertainAcc, width, color=colorsArray[1])# ='#188487')
        partialAcc = (0.9935, 1.08433734939759, 0.988888888888889, 1, 0.984932691, 1.011904762, 0.988505747, 0.943820225)
        rects3 = ax.bar(ind+2*width, partialAcc, width, color=colorsArray[2])# ='#7A68A6')
        stqbcAcc = (0.9935, 1.07228915662651, 1.03448275862069, 0.995616565, 0.924910461, 0.952380952, 1.011494253, 0.95505618)
        rects4 = ax.bar(ind+3*width, stqbcAcc, width, color=colorsArray[3])#, color='#467821')
        sentIntAcc = (0.973998345, 1.089638554, 1.028850575, 0.989041412, 0.97999259, 0.971785714, 0.973793103, 0.928202247)
        rects5 = ax.bar(ind+4*width, sentIntAcc, width, color=colorsArray[4])# 'y')
        sentPolAcc = (0.986999173, 1.06939759, 1.04, 0.997808282, 0.969988885, 0.99, 0.956896552, 0.928202247)
        rects6 = ax.bar(ind+5*width, sentPolAcc, width, color=colorsArray[5])# '#e59c16')
        sentDistAcc = (0.980498759, 1.082891566, 1.037816092, 1.004383435, 0.95998518, 0.965714286, 0.96816092, 0.928876404)
        rects7 = ax.bar(ind+6*width, sentDistAcc, width, color=colorsArray[6])# '#a06d0f')
        
        # add some text for labels, title and axes ticks
        #ax.set_ylabel('Relative Accuracy', color='k', fontsize=12)
        #ax.set_title('Scores by group and gender', color='k')
        ax.set_xticks(ind+2*width)
        ax.set_xticklabels( ('Similar', 'Different', 'Similar', 'Different', 'Similar', 'Different', 'Similar', 'Different') , color='k', fontsize=12)
        
        lg = ax.legend( (rects1[0], rects2[0], rects3[0], rects4[0], rects5[0], rects6[0], rects7[0]), ('Source SVM', 'Uncertainty Sampling', 'Partial QBC', 'Source & Target QBC', 'Sentiment Intensity', 'Senitment Polarity', 'Senitment Distinctness'),prop={'size':7.4} )
        lg.draw_frame(True)
        frame = lg.get_frame()
        frame.set_ec('0.45')
        frame.set_linewidth(1)
        
        #autolabel(rects1)
        #autolabel(rects2)
        #autolabel(rects2)
        #autolabel(rects2)
        plt.ylim(0.8,1.1)
        ax.axhline(y=1, linewidth=2, color='k', zorder=0, ls='dashed')
        plt.savefig(getFolderPath() + 'accuracy_bar_graph.pdf')
        plt.show()

def plotAlgAccuracy(sourceAcc, targetAcc, algAcc, filename, ylimlow = 0.5, ylimup = 1):
    with plt.style.context('fivethirtyeight'):
        
        #matplotlib.rc('axes', edgecolor='k', linewidth=0.5)    
        #matplotlib.rc('grid', color='k')
        
        ("Plotting "+filename)
        fig1 = plt.figure(1)#, edgecolor='k', linewidth=1)
        fig1.set_size_inches(9,6)
        #fig1.suptitle('Accuracy of Uncertainty Selection', fontsize=13)
        #fig1.patch.set_facecolor('w')
        acuPlt = fig1.add_subplot(111)#, axisbg='w')
        values = [0,20,50,100,150,200,250,300,350,400,450,500,550,600,650,700]
        acuPlt.plot(values, [sourceAcc for i in range(len(values))], '--', color='#d11141')
        acuPlt.plot(values, [targetAcc for i in range(len(values))], '--', color='#00B159')
        acuPlt.plot(values, algAcc, color='#00aedb')
        plt.ylim(ylimlow, ylimup)
        #uncAcuPlt.set_xlabel('Number of instances', fontsize=12, color='k')
        #uncAcuPlt.set_ylabel('Accuracy', fontsize=12, color='k')
        acuPlt.tick_params(axis='x', labelcolor='k', labelsize = 18)#, colors='k',)
        acuPlt.tick_params(axis='y', labelcolor='k', labelsize = 18)#, colors='k', size=20)
        #for spine in uncAcuPlt.spines.values():
        #    spine.set_edgecolor('k')
        plt.savefig(getFolderPath()+filename)
        plt.show()
        
def getNormalizedAccValues(accValues, baseValue, stretchFactor):
    return [(value-baseValue)*stretchFactor for value in accValues]
    
def plotComparativeGraph(xValues, valueArrays, labelsArray, colorsArray, fileName, ylowlim, yuplim, legendSize):
    with plt.style.context('fivethirtyeight'):

        fig1 = plt.figure(1)#, edgecolor='k', linewidth=1)
        fig1.set_size_inches(9,6)
        #fig1.suptitle('Accuracy of Uncertainty Selection', fontsize=13)
        #fig1.patch.set_facecolor('w')
        acuPlt = fig1.add_subplot(111)#, axisbg='w')
        handlesArray = [None]*len(valueArrays)
        i = 0
        for array in valueArrays:
            handlesArray[i], = acuPlt.plot(xValues, array, color = colorsArray[i])
            i += 1
        #uncAcuPlt.set_xlabel('Number of instances', fontsize=12, color='k')
        #uncAcuPlt.set_ylabel('Accuracy', fontsize=12, color='k')
        plt.ylim(ylowlim, yuplim)
        acuPlt.tick_params(axis='x', labelcolor='k', labelsize = 18)#, colors='k',)
        acuPlt.tick_params(axis='y', labelcolor='k', labelsize = 18)#, colors='k', size=20)
        #for spine in uncAcuPlt.spines.values():
        #    spine.set_edgecolor('k')
        
        #Legend    
        lg = fig1.legend( handles=handlesArray, labels = labelsArray, prop={'size':legendSize} ) 
        lg.draw_frame(True)
        frame = lg.get_frame()
        frame.set_ec('0.45')
        frame.set_linewidth(1)        
        
        plt.savefig(getFolderPath()+fileName)
        plt.show()

defSourceAcc = 0.7307
defTargetAcc = 0.8876

#Uncertainty plot parameters
UncAcu = [0.7307 ,0.7388, 0.7807, 0.7807, 0.7997, 0.8024, 0.824, 0.8234, 0.8376, 
          0.8477, 0.8518, 0.8484, 0.8545, 0.8572, 0.8633, 0.8707]
uncFilename = 'uncertainty_accuracy.pdf'

#Partial QBC plot parameters
partAcu = [0.7307, 0.7753721244925575,0.7848443843031123,0.7976995940460081,
           0.7861975642760487,0.8227334235453315,0.8369418132611637,
           0.8342354533152909,0.8328822733423545,0.8267929634641408,
           0.8572395128552097,0.8592692828146143,0.8633288227334236,
           0.8619756427604871,0.8599458728010826,0.8572395]
partFilename = 'partialQBC_accuracy.pdf'

#Source & Target QBC plot parameters
stAcu = [0.7307, 0.7713125845737483, 0.7821380243572396, 0.7760487144790257,
            0.7895805142083897,0.8328822733423545,0.8240866035182679,
           0.8247631935047361,0.8288227334235453,0.8410013531799729,
           0.8470906630581867,0.8491204330175913,0.8558863328822733,
           0.854533152909337,0.8660351826792964,0.8565629228687416]
stFilename = 'STQBC_accuracy.pdf'

sentSourceAcc = 0.8029556650246306
sentTargetAcc = 0.8735632183908046
sentylimlow = 0.6
senylimup = 0.9

#Sentiment Intensity plot parameters
sentIntAcu = [0.8029556650246306, 0.8111658456486043, 0.825944170771757, 0.8325123152709359,
            0.8505747126436781, 0.8472906403940886, 0.8472906403940886,
           0.8407224958949097, 0.8555008210180624, 0.8522167487684729,
           0.8587848932676518, 0.8571428571428571, 0.8637110016420362,
           0.8538587848932676, 0.8653530377668309, 0.8719211822660099]
sentIntFilename = 'sentiment_intensity_accuracy.pdf'

#Sentiment Polarity plot parameters
sentPolAcu = [0.8029556650246306, 0.8144499178981938, 0.8210180623973727, 0.8275862068965517, #0, 20, 50, 100
            0.8374384236453202, 0.8571428571428571, 0.8341543513957307, #150, 200, 250
           0.8620689655172413, 0.8653530377668309, 0.8604269293924466,  #300, 350, 400
           0.8735632183908046, 0.8669950738916257, 0.8686371100164204,  #450, 500, 550
           0.8801313628899836, 0.8686371100164204, 0.8719211822660099]  #600, 650, 700
sentPolFilename = 'sentiment_polarity_accuracy.pdf'

#Sentiment Distinctness plot parameters
sentDistAcu = [0.8029556650246306, 0.8357963875205254, 0.8325123152709359, 0.8440065681444991, #0, 20, 50, 100
            0.8555008210180624, 0.8472906403940886, 0.8423645320197044, #150, 200, 250
           0.8423645320197044, 0.8440065681444991, 0.8505747126436781,  #300, 350, 400
           0.8587848932676518, 0.8587848932676518, 0.8555008210180624,  #450, 500, 550
           0.8571428571428571, 0.8587848932676518, 0.8784893267651889]  #600, 650, 700
sentDistFilename = 'sentiment_distinctness_accuracy.pdf'


#plotAlgAccuracy(defSourceAcc, defTargetAcc, UncAcu, uncFilename)
#plotAlgAccuracy(defSourceAcc, defTargetAcc, partAcu, partFilename)
#plotAlgAccuracy(defSourceAcc, defTargetAcc, stAcu, stFilename)
#plotAlgAccuracy(sentSourceAcc, sentTargetAcc, sentIntAcu, sentIntFilename, sentylimlow, senylimup)
#plotAlgAccuracy(sentSourceAcc, sentTargetAcc, sentPolAcu, sentPolFilename, sentylimlow, senylimup)
#plotAlgAccuracy(sentSourceAcc, sentTargetAcc, sentDistAcu, sentDistFilename, sentylimlow, senylimup)

orgDiff = defTargetAcc - defSourceAcc
sentDiff = sentTargetAcc - sentSourceAcc
orgStrech = 1/orgDiff
sentStrech = 1/sentDiff

normUncAcu = getNormalizedAccValues(UncAcu, defSourceAcc, orgStrech)
normPartAcu = getNormalizedAccValues(partAcu, defSourceAcc, orgStrech)
normStAcu = getNormalizedAccValues(stAcu, defSourceAcc, orgStrech)

normSentIntAcu = getNormalizedAccValues(sentIntAcu, sentSourceAcc, sentStrech)
normSentPolAcu = getNormalizedAccValues(sentPolAcu, sentSourceAcc, sentStrech)
normSentDistAcu = getNormalizedAccValues(sentDistAcu, sentSourceAcc, sentStrech)

#print("original sentDistAcu:")
#print(sentDistAcu)
#normSentDistAcu = getNormalizedAccValues(sentDistAcu, sentSourceAcc, sentStrech)
#print("normalized sentDistAcu:")
#print(normSentDistAcu)

accXvalues = [0,20,50,100,150,200,250,300,350,400,450,500,550,600,650,700]

valArray = [normUncAcu, normPartAcu, normStAcu, normSentIntAcu, normSentPolAcu, normSentDistAcu]
#colorsArray = ['#3b5998', '#adff00', '#74d600', '#028900', '#fe8181', '#fe2e2e', '#b62020'] # blue for source, greens for general, reds for sentiment
colorsArray = ['#00aedb', '#4cc88a', '#00B159', '#007b3e', '#de587a', '#d11141', '#920b2d'] # Metro UI-like pastels. blue for source, greens for general, reds for sentiment
#colorsArray = ['#dea666', '#86dd9f', '#53CF77', '#3a9053', '#d08f8f', '#bc5f5f', '#834242'] # pastels. blue for source, greens for general, reds for sentiment
compAccFileName = 'comparative_accuracy.pdf'
compAccLabelsArray = ['Uncertainty Sampling', 'Partial QBC', 'Source & Target QBC', 'Sentiment Intensity', 'Senitment Polarity', 'Senitment Distinctness']
#plotComparativeGraph(accXvalues, valArray, compAccLabelsArray, colorsArray[1:], compAccFileName, 0, 1.5, 13)

#Plotting comparative accuracy for the basic simulation
klXvalues = [0, 12827, 26056, 42532, 51172, 67003, 69048, 94916, 103604, 125715]
simSourceAcc = [0.688, 0.627, 0.6425, 0.682, 0.55, 0.589, 0.573, 0.568, 0.6475, 0.515]
simTargetAcc = [0.693, 0.666, 0.658, 0.606, 0.717, 0.664, 0.683, 0.728, 0.613, 0.793]
simUncAcc = [0.696, 0.657, 0.672, 0.732, 0.721, 0.738, 0.73, 0.75, 0.722, 0.712]
simCompAccFileName = 'sim_comparative_accuracy.pdf'
simCompAccLabelsArray = ['Source SVM', 'Target SVM', 'Uncertainty Sampling']
plotComparativeGraph(klXvalues, [simSourceAcc, simTargetAcc, simUncAcc], simCompAccLabelsArray, [colorsArray[5], colorsArray[2], colorsArray[0]], simCompAccFileName, 0.5, 0.85, 18)

#plotAccuracyBarGraph(colorsArray)
        
print("finished plotting")



