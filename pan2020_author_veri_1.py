import sys
sys.path.append("/anaconda3/lib/python3.7/site-packages")
import os
import re, string, unicodedata
import nltk
import scipy
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
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk import download
download('stopwords')  # Download stopwords list.
stop_words = stopwords.words('english')

print ("Authorship Verification loaded")

def print_most_frequent(ngrams, num=10):
    """Print num most common n-grams of each length in n-grams dict."""
    for n in sorted(ngrams):
        print('----- {} most common {}-grams -----'.format(num, n))
        for gram, count in ngrams[n].most_common(num):
            print('{0}: {1}'.format(' '.join(gram), count))
        print('')


def posNgrams(s,n):
    '''Calculate POS n-grams and return a dictionary'''
    text = nltk.word_tokenize(s)
    text_tags = nltk.pos_tag(text)
    taglist = []
    output = {}
    for item in text_tags: 
        taglist.append(item[1])
    for i in range(len(taglist)-n+1):
        g = ' '.join(taglist[i:i+n])
        output.setdefault(g,0)
        output[g] += 1
    return output

def n_grams(sentence, n):
    return ngrams(word_tokenize(sentence), n)

def POS_tag(sentence):
    tokens=nltk.word_tokenize(sentence)
    return nltk.pos_tags(tokens)

def preprocess(sentence):
    return [w for w in sentence.lower().split() if w not in stop_words]


def WordFeatures1(word_list, all_training_text):
    fvs_words = np.array([[all_training_text.count(word)/(len(all_training_text)) for word in word_list]])
    return fvs_words

def Authorship_Verification (inputFolder, outputFolder):
    
    answers=[]
    EVALUATION_DIRECTORY = "/Users/catherine/Downloads/pan20-authorship-verification-training-small"

    with open(EVALUATION_DIRECTORY +os.sep+ "pan20-authorship-verification-training-small.jsonl",'r') as json_file:
        #for line in reversed(json_file.readlines()):
        for line in (json_file.readlines()):
             data= json.loads(line)
             ID=((data)['id'])
             ss1=(((data)['pair'])[0])
             ss2=(((data)['pair'])[1])
            
             n=6
             fdist_ss1=[word for word, word_count in (nltk.FreqDist(posNgrams(ss1, n))).most_common(100)]
             fdist_ss2= [word for word, word_count in (nltk.FreqDist(posNgrams(ss2, n))).most_common(100)]
             
             ss1_pos= [word for word, word_count in (posNgrams(ss1, n)).items()]
             ss2_pos= [word for word, word_count in (posNgrams(ss2, n)).items()]

             fdist_ss=fdist_ss1 + fdist_ss2 #+list(set(preprocess(ss1)) - set(preprocess(ss2)))
             
             sentence_1_avg_vector=WordFeatures1(fdist_ss, (ss1_pos))
             
             sentence_2_avg_vector=WordFeatures1(fdist_ss, (ss2_pos))
             
##             sentence_1_avg_vector=np.array([word_count for word, word_count in (nltk.FreqDist(posNgrams(ss1, 3))).most_common(1000)])
##             sentence_2_avg_vector=np.array([word_count for word, word_count in (nltk.FreqDist(posNgrams(ss2, 3))).most_common(1000)])
##             
            
             cosine =  cosine_similarity(sentence_1_avg_vector,sentence_2_avg_vector)
             print('id', ID,'value', cosine[0][0])
             answers.append({'id': ID,'value': cosine[0][0]})

    OUTPUT_DIRECTORY="/Users/catherine/Downloads/kocher16-master/output/new1"
    with open(OUTPUT_DIRECTORY+os.sep+'answers3.jsonl', 'w') as outfile:
         for ans in answers:
             json.dump(ans, outfile)
             outfile.write('\n')
           
    return()

if __name__ == '__main__':
         print("Authorship Verification ")
         inputFolder = "/Users/catherine/Downloads/pan20-authorship-verification-training-small"
         outputFolder="/Users/catherine/Downloads/kocher16-master/output/new1"
         Authorship_Verification (inputFolder, outputFolder)
