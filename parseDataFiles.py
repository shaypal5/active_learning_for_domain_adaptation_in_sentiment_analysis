import numpy as np
import re
import string

def parseDataFiles(dataFile, fileLabel, convert2Lower):
#fileLabel is the Label of all the data in dataFile (i.e - positive = 1/negative = -1)
    file = open(dataFile)
    data = []
    line = file.readline()
    while line:
        isTag, strippedLine = stripLine(line);
        if strippedLine == 'review':
            review = {}
            review['label'] = fileLabel
            review = parseInnerText(strippedLine,file,review)
            if convert2Lower:
                review['review_text'] = review['review_text'].lower()
            #split to words and remove punctuation marks
            remove_punct_map = dict.fromkeys(map(ord, string.punctuation))
            print(review)
            punctLess = review['review_text'].translate(remove_punct_map)
            review['review_words'] = punctLess.split();
            data.append(review)
        line = file.readline()
    print('end of file')
    return data
    
def parseInnerText(tag,file,review):
    value = ''
    line = file.readline()
    isTag, strippedLine = stripLine(line)
    while line and not (strippedLine == tag): #check if this is the end tag of the open "tag"
        if isTag: #this is a new opening tag
            review = parseInnerText(strippedLine,file,review)
        else:
            value = value + strippedLine
        line = file.readline()
        isTag, strippedLine = stripLine(line)
    review[tag] = value.strip()
    return review
    
def stripLine(line):
# Checks if a line is a tag line.
# Returns the original line if not, and the tag if it is. 
    lineTag = line
    isTag = False
    regExp = '<\/?([a-zA-Z_]*)>'
    match = re.match(regExp,line)
    if match:
        print(match)
        isTag = True
        lineTag = match.group(1)
    return (isTag,lineTag)
    