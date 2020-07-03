import sys
sys.path.append("/anaconda3/lib/python3.7/site-packages")
import os
import re, string, unicodedata
import nltk
import scipy
import csv
from bs4 import BeautifulSoup
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
import jsonlines
import json
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
from collections import Counter
import operator
from sklearn.metrics.pairwise import cosine_similarity
import math
import numpy as np
import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from nltk import download
download('stopwords')  # Download stopwords list.
stop_words = stopwords.words('english')

print ("Authorship Verification loaded")


def preprocess(sentence):
    #return [w for w in sentence.lower().split() if w not in stop_words]
    return [w for w in sentence.lower().split()]

def combine_text(sentence):
    return [w for w in sentence.lower().split()]


def WordFeatures1(word_list, all_training_text):
    fvs_words = np.array([[all_training_text.count(word)/(len(all_training_text)) for word in word_list]])
    return fvs_words



txt_files =[ "/Users/catherine/Downloads/pan20-authorship-verification-training-small/pan20-authorship-verification1.csv"]
for txt_file in txt_files:
    with open(txt_file, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=",")
        next(reader) # skip header
        word_list =  [r[0] for r in reader]

        
def Authorship_Verification (inputFolder, outputFolder):
  
    answers=[]
    EVALUATION_DIRECTORY = inputFolder

    #with open(EVALUATION_DIRECTORY +os.sep+ "pairs.jsonl",'r') as json_file:
    with open(EVALUATION_DIRECTORY +os.sep+ "pan20-authorship-verification-training-small.jsonl",'r') as json_file:
        for line in (json_file.readlines()):
             data= json.loads(line)
             ID=((data)['id'])
             ss1=(((data)['pair'])[0])
             ss2=(((data)['pair'])[1])
             ss1=re.findall(r"[\w']+|[.,!?;]", ss1)
             ss1=[j.lower() for j in ss1]
             ss2=re.findall(r"[\w']+|[.,!?;]", ss2)
             ss2=[j.lower() for j in ss2]
             #print(ss2[0:10])
             
             j=500
             #fdist_ss=word_list[0:j]
             fdist_ss1 = nltk.FreqDist(ss1)
             fdist_ss1 = sorted(dict(fdist_ss1).items(), key=operator.itemgetter(1), reverse=True)
             fdist_ss1=[x[0] for x in fdist_ss1 ]
             
             fdist_ss2 = nltk.FreqDist(ss2)
             fdist_ss2 = sorted(dict(fdist_ss2).items(), key=operator.itemgetter(1), reverse=True)
             fdist_ss2=[x[0] for x in fdist_ss2 ]
             
             fdist_ss=fdist_ss1[0:j] + fdist_ss2[0:j]
             
             sentence_1_avg_vector=WordFeatures1(fdist_ss, (ss1))
             sentence_2_avg_vector=WordFeatures1(fdist_ss, (ss2))
             
             cosine =  cosine_similarity(sentence_1_avg_vector,sentence_2_avg_vector)
             cosine=round((cosine[0][0]),4)
            
             answers.append({'id': ID,'value': cosine})

    OUTPUT_DIRECTORY=outputFolder
    with open(OUTPUT_DIRECTORY+os.sep+'answers'+str(j)+'.3.jsonl', 'w') as outfile:
         for ans in answers:
             json.dump(ans, outfile)
             outfile.write('\n')
           
    return()

if __name__ == '__main__':
         print("Authorship Verification ")
         
