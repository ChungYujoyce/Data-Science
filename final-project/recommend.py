import nltk,os
import requests
import re
import pandas as pd
nltk.download('punkt')
import numpy as np

def tf(term, token_doc):
    tf = token_doc.count(term)/len(token_doc)
    return tf

# create function to calculate how many doc contain the term 
def numDocsContaining(word, token_doclist):
    doccount = 0
    for doc_token in token_doclist:
        if doc_token.count(word) > 0:
            doccount +=1
    return doccount
  
import math
# create function to calculate  Inverse Document Frequency in doclist - this list of all documents
def idf(word, token_doclist):
    n = len(token_doclist)
    df = numDocsContaining(word, token_doclist)
    if df==0:
        return 0
    else:
        return math.log10(n/df)

#define a function to do cosine normalization a data dictionary
def cos_norm(dic): # dic is distionary data structure
    dic_norm={}
    factor=1.0/np.sqrt(sum([np.square(i) for i in dic.values()]))
    for k in dic:
        dic_norm[k] = dic[k]*factor
    return dic_norm

#create function to calculate normalize tfidf 
def compute_tfidf(token_doc,bag_words_idf):
    tfidf_doc={}
    for word in set(token_doc):
        if word not in bag_words_idf.keys(): # may not find keys
            pass
        else:
            tfidf_doc[word]= tf(word,token_doc) * bag_words_idf[word] 
    tfidf_norm = cos_norm(tfidf_doc)
    return tfidf_norm

# create normalize term frequency
def tf_norm(token_doc):
    tf_norm={}
    for term in token_doc:
        tf = token_doc.count(term)/len(token_doc)
        tf_norm[term]=tf
    tf_max = max(tf_norm.values())
    for term, value in tf_norm.items():
        tf_norm[term]= 0.5 + 0.5*value/tf_max
    return tf_norm

def compute_tfidf_query(query_token,bag_words_idf):
    tfidf_query={}
    tf_norm_query = tf_norm(query_token)
    for term, value in tf_norm_query.items():
        tfidf_query[term]=value*bag_words_idf[term]   
    return tfidf_query

import pickle, time, os
from datetime import datetime

path_df = "./Pickles/News_dataset.pickle"

with open(path_df, 'rb') as data:
    df = pickle.load(data)


doc_all = {}

for i in range(100):
    doc_all[df.loc[i]['Title']] = df.loc[i]['Content'].split()
    
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
start = time.time()
#create bag words
bag_words =[] # declare bag_words is a list
for doc in doc_all.keys():
    bag_words += doc_all[doc]
bag_words=set(bag_words)
#calculate idf for every word in bag_words
bag_words_idf={}  
bag_words_len = len(bag_words)
bag_word_10 = round(bag_words_len/10,0)
print("the number of term in bag_word", bag_words_len)

i=0
for word in bag_words:
    i+=1
    if (i%bag_word_10==0):print("finish %s idf processing" %(str(round(i*10/bag_word_10))+"%"))
    bag_words_idf[word]= idf(word,doc_all.values())

##calculate tfidf with cosine normalization
tfidf={} # declare tfidf dictionary to store tfidf value
for doc in doc_all.keys():
    tfidf[doc]= compute_tfidf(doc_all[doc],bag_words_idf)

query="encourage wise hope"
query_token_raw= nltk.word_tokenize(query)
query_token = [term for term in query_token_raw if term in bag_words]

tfidf_query =compute_tfidf_query(query_token,bag_words_idf) #calculate tfidf for query text
print(tfidf_query)

# add tfidf of query text to tfidf of all doc and convert to dataframe
tfidf["query"] = tfidf_query

tfidf_df = pd.DataFrame(tfidf).transpose()
tfidf_df= tfidf_df.fillna(0) # replace all NaN by zero

from scipy.spatial.distance import cosine
cosine_sim ={}
for row in tfidf_df.index:
    if row != "query":
        cosine_sim[row]= 1-cosine(tfidf_df.loc[row],tfidf_df.loc["query"])

# the top 10 relevant document
cosine_sim_top10 = dict(sorted(cosine_sim.items(), key=lambda item: item[1],reverse=True)[:10])
print(cosine_sim_top10)

current_time = now.strftime("%H:%M:%S")
print("Finish tfidf processing at", current_time)
spent = time.time() - start
print("\nTotal spent time: "+str(spent) +"sec\n")

#plot barchart
import matplotlib.pyplot as plt
data = cosine_sim_top10
plt.barh(range(len(data)), list(data.values()), align='center', alpha=0.8)
plt.yticks(range(len(data)), list(data.keys())) # label for y axis
plt.xlabel('Smimilarity score')
plt.ylabel('news')

# save graph

plt.savefig("./recommand_barchart.png", bbox_inches='tight', dpi=600)
plt.show()