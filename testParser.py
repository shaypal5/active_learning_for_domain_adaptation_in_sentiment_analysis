# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 14:12:04 2014

@author: Shay
"""

import sys
import parseDataFiles
import parseProcessedDataFile
import parseProcessedDataFileForScikit
import nltk.classify
from sklearn.svm import LinearSVC
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder

folderPath = "C:/OrZuk/Data/sorted_data/apparel/"
balancedFileName = "processed.review.balanced"
allFileName = "processed.review"
trainFileName = "processed.review.trainset"
testFileName = "processed.review.testset"


#test parseDataFiles
def parseDataFiles():
    parsedData = parseDataFiles.parseDataFiles("C:/OrZuk/Data/sorted_data/apparel/positive.review",1,False)
    print(parsedData)
    print("Done testing parseDataFiles.py")
    stripped = parseDataFiles.stripLine("<review><unique_id>B0007QCQA4:good_sneakers:christopher_w._damico_")
    print(stripped)

#test parseProcessedDataFile.parseLine()
def testParseProcessedDataFileParseLine():
    parsedLine = parseProcessedDataFileForScikit.parseLine("friendly_but:1 not_leather:1 duped_the:1 sued:1 base:1 to_use:1 shipped:1 itself_was:1 smooth:1 my:1 card:1 book_but:1 doesn't_fit:1 was_user:1 disappointed:1 is_not:1 was_of:1 that's_only:1 was:4 wallet_i:1 product_but:1 it_was:2 duped:1 want_to:1 wallet_seem:1 very:4 cheap_the:1 like_it:1 no:1 only_the:1 cant_really:1 you:1 have_extreme:1 too_much:1 included_a:1 really:1 small_note:1 the_wallet:6 but_now:1 lining:1 quality_and:1 put_too:1 you_cant:1 doesn't:1 feel:1 higher:1 flare:1 excited_about:1 the_inside:1 material:1 transaction_itself:1 itself:2 fred_flare:1 a_small:1 snug_and:1 no_place:1 for_my:1 wallet_included:1 really_put:1 nylon:1 cant:1 saying_fred:1 leather:1 use_you:1 sued_but:1 is_no:1 sort_of:1 now_i:1 it's_not:1 very_excited:1 by_saying:1 of_looks:1 much:1 also:2 is_very:1 however_have:1 small:1 higher_quality:1 base_of:1 made:1 its_nylon:1 the_base:1 place:1 put:1 do_however:1 also_the:1 i.d._the:1 little:1 sort:1 material_is:1 part:1 slots_are:1 flare_shipped:1 leather_its:1 i.d.:1 only:1 want:1 it_also:1 wallet_the:1 about_this:1 shipped_this:1 the_pictures:1 use:1 however:1 note:1 i_feel:1 note_book:1 card_slots:1 hard_to:1 recommend:1 pictures_made:1 extreme_problems:1 fast_and:1 start_by:1 it_sort:1 fit:1 much_in:1 feel_duped:1 inside_material:1 saying:1 place_for:1 the_transaction:1 the_credit:1 wallet:6 book:1 start:1 little_too:1 looks:1 problems_with:1 user_friendly:1 too:2 quality:1 also_doesn't:1 snug:1 nylon_and:1 inside:1 i_do:2 the_lining:1 included:1 i:5 about:1 part_is:1 now:1 not:3 product_itself:1 to_start:1 itself_the:1 credit:1 recommend_this:1 wallet_also:1 fred:1 product_is:1 my_i.d.:1 made_the:1 of_higher:1 transaction:1 fast:1 excited:1 product_very:1 a_little:1 looks_cheap:1 i_want:1 lining_for:1 hard:1 very_fast:1 seem_like:1 is_sued:1 cheap:1 like:1 problems:1 i_was:1 was_very:2 wallet_part:1 friendly:1 very_hard:1 pictures:1 not_recommend:1 fit_in:1 user:1 too_snug:1 not_i:1 do_not:1 slots:1 the_product:2 product:5 this_product:3 credit_card:1 seem:1 extreme:1 smooth_i:1 very_smooth:1 #label#:negative")
    print(parsedLine)

#test parseProcessedDataFile
def testParseProcessedDataFile():
    print("==============================================================")
    print("Classifying with NLTK, so only feature presence (no word frequencies)")
    # Create a labelled feature set, for training
    trainSet = parseProcessedDataFile.parseDataFile(folderPath+trainFileName)
    testSet = parseProcessedDataFile.parseDataFile(folderPath+testFileName)
    print("Done parsing data")
    
    # Train the classifier
    classifier = nltk.classify.SklearnClassifier(LinearSVC())
    classifier.train(trainSet)

    successNum = 0
    failsNum = 0

    # Predict and compare to true labels
    for test_record in testSet:
        features, label = test_record
        predict = classifier.classify(features)
        #print("True label is "+str(label)+". Predicted "+str(predict))
        if predict == label:
            successNum += 1
        else:
            failsNum +=1
        
    print(str(successNum)+" successeses")
    print(str(failsNum)+" failures")

def checkScikitVectorizingFrequencyDicts():
    print("==============================================================")
    vec1 = {'jack': 4, 'sape': 2, 'is':23, 'it':19, 'monday':1, 'sunday':0}
    vec2 = {'is':15, 'it':13, 'funny':3, 'jack':4}
    data = [vec1,vec2]
    print(data)
    vectorizer = DictVectorizer(dtype=float, sparse=True)
    newData = vectorizer.fit_transform(data)
    print(newData[0])
    

# Using scikit directly
def checkScikitBasicFunctions():
    print("==============================================================")
    print("Trying to classify directly with scikit, with word frequencies")
    trainX, trainY  = parseProcessedDataFileForScikit.parseDataFile(folderPath+trainFileName)
    testX, testY = parseProcessedDataFileForScikit.parseDataFile(folderPath+testFileName)
    vectorizer = DictVectorizer(dtype=float, sparse=True)
    encoder = LabelEncoder()
    
    #print(trainX[0])
    #for x in trainX:
        #print(x)
    
    #vectorize!
    newTrainX = vectorizer.fit_transform(trainX)
    newTrainY = encoder.fit_transform(trainY)
    #print(newTrainX[0])
    #print(type(newTrainX[0]))
    
    #fit/train the linear SVM
    svc = LinearSVC()
    svc.fit(newTrainX, newTrainY)

    #predict
    newTestX = vectorizer.transform(testX)
    classes = encoder.classes_
    #predictions = svc.predict(newTestX)
    correct = 0
    wrong = 0
    for i in range(len(testX)):
        prediction = svc.predict(newTestX[i])
        if classes[prediction] == testY[i]:
            correct += 1
        else:
            wrong += 1
    
    print("Got "+str(correct)+" correct.")
    print("Got "+str(wrong)+" wrong.")
    
# Running the tests
checkScikitVectorizingFrequencyDicts()
checkScikitBasicFunctions()
testParseProcessedDataFile()