# -*- coding: utf-8 -*-

import os

from collections import Counter

print ("Read WordDict.py")


def dictFromFile(aFileName):
    with open(aFileName) as inFile:
        listsOfWords = inFile.read().split()
    return Counter(listsOfWords)


def getListListPANAndFoldersPAN(aPath):
    aListListPAN = []
    foldersPAN = []
    
    for d in os.listdir(aPath):#list of directories in the data folder
        
        aListPAN = []
        if os.path.isfile(aPath + "/" + d):
            continue
        foldersPAN.append(d)
        for f in os.listdir(aPath + "/" + d + "/"):#text files
            aListPAN.append(aPath + "/" + d + "/" + f)#text files and thier path
        aListListPAN.append(sorted(aListPAN))
    foldersPAN, aListListPAN = zip(*sorted(zip(foldersPAN, aListListPAN)))
    return aListListPAN, foldersPAN
