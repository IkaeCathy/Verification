import sys
sys.path.append("/anaconda3/lib/python3.7/site-packages")
import os
import re, string, unicodedata
import nltk
import contractions
import inflect
import scipy
from bs4 import BeautifulSoup
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
import jsonlines
import json
from nltk.tokenize import word_tokenize
import gensim
from nltk.corpus import stopwords
from nltk import download
#download('stopwords')  # Download stopwords list.
stop_words = stopwords.words('english')
#if sys.version_info[0] >= 3:
    #unicode = str
#with jsonlines.open("/Users/catherine/Downloads/pan20-authorship-verification-training-small/pan20-authorship-verification-training-small.jsonl") as f:
    #for line in f.iter():
        #print(line)


##with open("/Users/catherine/Downloads/pan20-authorship-verification-training-small/pan20-authorship-verification-training-small-truth.jsonl",'r') as json_file:
##    for line in json_file.readlines():
##         data= json.loads(line)
##         print(len(data))
##         print((data))
##         for item, value in data:
##             print(item, value)
def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def normalize(words):
    #print(words[0:10])
    words = nltk.word_tokenize(words)
    #words = remove_non_ascii(words)
    words = to_lowercase(words)
    #words = lemmatize_verbs(words)
    #words = stem_words(words)
    #words = remove_punctuation(words)
    #words = replace_numbers(words)
    #words = remove_stopwords(words)
    
    return words


import numpy as np

import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('punkt') # if necessary...
stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]
'''remove punctuation, lowercase, stem'''
def normalize(text):
    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))
vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')
def cosine_sim(text1, text2):
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0,1]


def loadGloveModel(File):
    print("Loading Glove Model")
    f = open(File,'r')
    gloveModel = {}
    for line in f:
        splitLines = line.split()
        word = splitLines[0]
        wordEmbedding = np.array([float(value) for value in splitLines[1:]])
        gloveModel[word] = wordEmbedding
    print(len(gloveModel)," words loaded!")
    return gloveModel

#gloveFile=('/Users/catherine/Downloads/glove.6B.100d.txt')
#model=loadGloveModel(gloveFile)
from gensim.models import FastText



def cosine_distance_wordembedding_method(s1, s2):
    #vector_1 = np.mean([model[word] for word in normalize(s1)],axis=0)
    #vector_2 = np.mean([model[word] for word in normalize(s2)],axis=0)
    model1 = FastText(normalize(s1), size=128)
    model2 = FastText(normalize(s2), size=128)
    vector_1 = np.mean([model1[word] for word in normalize(s1)],axis=0)
    vector_2 = np.mean([model2[word] for word in normalize(s2)],axis=0)
    cosine = scipy.spatial.distance.cosine(vector_1, vector_2)
    cosine=round((1-cosine),4)
    return cosine
    #print('Word Embedding method with a cosine distance asses that our two sentences are similar to',round((1-cosine)*100,2),'%')


#load word2vec model, here GoogleNews is used
#modelwm = gensim.models.KeyedVectors.load_word2vec_format('/Users/catherine/Downloads/GoogleNews-vectors-negative300.bin', binary=True)
#two sample sentences 
#calculate distance between two sentences using WMD algorithm
#from keras.preprocessing.text import Tokenizer
#vocab_size = 100
#tokenizer = Tokenizer(num_words=vocab_size)#IF I USE OTHER TOKENIZER...
#sentence_1 = "Today is very cold."  
#sentence_2 = "I'd like something to drink."    
#print(modelwm.wv.wmdistance(sentence_1.split(" "), sentence_2.split(" ")))

def preprocess(sentence):
    return [w for w in sentence.lower().split() if w not in stop_words]

from gensim.corpora.dictionary import Dictionary
answers=[]
with open("/Users/catherine/Downloads/pan20-authorship-verification-training-small/pan20-authorship-verification-training-small.jsonl",'r') as json_file:
    #for line in (reversed(json_file.readlines())):
    for line in (json_file.readlines())[0:10]:
         data= json.loads(line)
         ID=((data)['id'])
         #print((data)['fandoms'])
         ss1=(((data)['pair'])[0])
         ss2=(((data)['pair'])[1])
         cosine=cosine_distance_wordembedding_method((ss1), (ss2))
         print("ID:", ID, "cosine:",cosine)
         answers.append({'id': ID,'value': cosine})
         
output_folder="/Users/catherine/Downloads/kocher16-master/output/new"
with open(output_folder+os.sep+'answers.jsonl', 'w') as outfile:
     for ans in answers:
         json.dump(ans, outfile)
         outfile.write('\n')
       
