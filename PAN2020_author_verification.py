#['"Yes, ther', 'Now that t', 'Tracing an', 'One day tw', 'Gackt and ', 'Peace Make', 'Munder ben', 'Olivia wal', 'Axel looke', 'With a dep']

import sys
sys.path.append("/anaconda3/lib/python3.7/site-packages")
import os
import jsonlines
import json
import nltk
import math
import operator
import re
import pandas as pd

def preprocess(sentence):
    #return [w for w in sentence.lower().split() if w not in stop_words]
    return [w for w in sentence.lower()]

pairs=[]
inputFolder="/Users/catherine/Downloads/pan20-authorship-verification-training-small"
EVALUATION_DIRECTORY = inputFolder
with open(EVALUATION_DIRECTORY +os.sep+ "pan20-authorship-verification-training-small.jsonl",'r') as json_file:
    for line in (json_file.readlines()):
            data= json.loads(line)
            ss1=(((data)['pair'])[0])
            ss2=(((data)['pair'])[1])
            #print(ss1[0:10], ss2[0:10])
            pairs.append(ss1)
            pairs.append(ss2)
x=(list(set(pairs)))#take only a text line once
print(x[0])
#y=[re.findall(r"[\w']+|[.,!?;]", i) for i in x]
#print(len(y))
#z=[j.lower() for i in y for j in i]
#print(z[0:10])
#fdist_z = nltk.FreqDist(z)
#fdist_z = sorted(dict(fdist_z).items(), key=operator.itemgetter(1), reverse=True)
#fdist_z = [[(fdist_z)[j][0] , (fdist_z[j][1])  ] for j in range(len(fdist_z)) if (fdist_z)[j][1] > 0]
#df = pd.DataFrame(fdist_z, columns = [ 'Term', 'Term_Freq'])
#df.to_csv('/Users/catherine/Downloads/pan20-authorship-verification-training-small/pan20-authorship-verification1.csv', index=False, encoding='utf-8',sep='\t')  

#with open("/Users/catherine/Downloads/pan20-authorship-verification-training-small/pan20-authorship-verification1.txt", "w") as text_file:
    #for s in fdist_z:
        #text_file.write(str(s[0]) + '\t'+ str(s[1]) + '\n')
