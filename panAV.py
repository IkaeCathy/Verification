# -*- coding: utf-8 -*-

import getopt
import random
import sys
import time

from evaluator import writeNameAndScore, normProbs
from Spatium import featureSelection, mergeToProfile, l1Distance
from WordDict import dictFromFile, getListListPANAndFoldersPAN

print ("Read panAV.py")

compNo = 5.0  # compare X times
randCompare = 3  # compare with Y random candidates
wLen = 200

random.seed(1811)

#handles several text files
def processAll(aListOfFile):
    wLists = []
    for aFile in aListOfFile:
        texts = []
        for aText in aFile:
            texts.append(dictFromFile(aText))
        wLists.append(texts)
    return wLists

#for one text file
def processSamples(aListOfSamples):
    wLists = []
    for aSample in aListOfSamples:
        wLists.append(dictFromFile(aSample))
    return wLists


def runIt():
    print (len(allKnownDocs))

    myProbs = []
    myNewProbs = []
    t = 0.025
    for i in range(len(allKnownDocs)):
        #print (i, foldersPAN[i])#lists all folders in the data directory
        res = []
        aWordList = featureSelection(allUnknownDocs[i], wLen)#extract wLen features from each document
        print(aWordList)
        for _ in range(int(compNo)):
            docs = [allKnownDocs[i]]
            choice = list(range(len(aListListPAN)))
            
            del choice[i]
            random.shuffle(choice)
            notChosen = choice[:]
            for r in choice:
                if (len(allKnownDocs[r]) > 0 and
                        len(aListListPAN[r]) == len(aListListPAN[i])):
                    docs.append(allKnownDocs[r])
                    del notChosen[notChosen.index(r)]
            docs = docs[:randCompare+1]
            r = 0
            while len(docs) <= randCompare:
                docs.append(allKnownDocs[notChosen[r]])
                r += 1

            profiles = mergeToProfile(docs)
            d = l1Distance(profiles, allUnknownDocs[i], aWordList)

            dd = d[:]
            dd[0] = float("inf")
            if min(d) == d[0]:
                if min(dd)/d[0] > 1.0+t:
                    res.append(round(min(dd)/d[0], 2))
                else:
                    res.append(1.0)
            elif min(d)/d[0] < 1.0-t and min(d) != 0:
                res.append(round(min(d)/d[0], 2))
            else:
                res.append(1.0)

        myProbs.append(sum(res)/compNo)

    myProbs = normProbs(myProbs, t)
    aName = "answers"
    writeNameAndScore(foldersPAN, myProbs, outputFolder, aName)


if __name__ == '__main__':
    print ("The module panAV.py was uploaded\n")
    startTime = time.clock()

    inputFolder = ""
    outputFolder = ""

    try:
        opts, args = getopt.getopt(sys.argv[1:], "i:o:")
    except getopt.GetoptError:
        print ("panAV.py -i <inputFolder> -o <outputFolder>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-i":
            inputFolder = arg
        elif opt == "-o":
            outputFolder = arg

    assert len(inputFolder) > 0
    print ("Input folder is", inputFolder)
    assert len(outputFolder) > 0
    print ("Output folder is", outputFolder)

    aListListPAN, foldersPAN = getListListPANAndFoldersPAN(inputFolder)

    allKnownDocs = processAll([x[:-1] for x in aListListPAN])
    allUnknownDocs = processSamples([x[-1] for x in aListListPAN])

    runIt()

    print ("\n done in %.2fs" % (time.clock() - startTime))
