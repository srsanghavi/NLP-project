
# coding: utf-8

# <h2>Import Dependencies</h2>

# In[2]:


import urllib
from bs4 import BeautifulSoup as bs
import nltk
import pandas  as pd
import numpy as np
from nltk.stem.snowball import SnowballStemmer
from nltk import ngrams
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
from nltk.cluster.util import cosine_distance
from gensim.models import Word2Vec
import numpy as np
from nltk.tokenize import RegexpTokenizer
from operator import itemgetter 


# <h2> Parse and clean data </h2>

# In[3]:


def clean_data(html,baseurl):
    soup = bs(html, 'html.parser')
    if baseurl == 'i':
        entries = soup.find_all('div',{'class':'description'})
    elif baseurl == 't':
        entries = soup.find_all('div',{'itemprop':'articleBody'})
    elif baseurl == 'h':
        entries = soup.find_all('div',{'itemprop':'articlebody'})
    for each in entries:
            if each.figure:
                each.figure.decompose()
    content = []
    for e in entries:
         content.extend(e.find_all("p"))
    
    text = ""
    for each in content:
        text = text + each.get_text() +" "

    text = text.encode('utf-8').decode("unicode_escape").encode('ascii','ignore')
    text = nltk.sent_tokenize(text)    
    return text


# <h2> Luhn's Approach </h2>
# <h3> Extraction </h3>

# In[4]:


def word_scores(text):
    # tokenize the article
    tokenizer = RegexpTokenizer(r'\w+')
    word_list=[]
    for t in text:
        word_list.extend(tokenizer.tokenize(t))


    # remove stop words from the article
    filtered_words = [word for word in word_list if word.lower() not in stopwords.words('english')]
    total_words = len(filtered_words)
    words = filtered_words
    # find n-gram probability for filtered words
    filtered_words  = []
    for each in Counter(words).items():
        filtered_words.append([each[0] , float(each[1])/float(total_words)])

    # identify important words
    important_words = []
    for item in filtered_words:
        if item[1]>0.003:
            important_words.append(item[0])

    while '. ' in text:
        text = text.replace('. ','\n') 

    while "\n" in text:
        text = text.split("\n")
        
    vect = TfidfVectorizer(stop_words='english')
    dtm = vect.fit_transform(text)
    features = vect.get_feature_names()
    
    scores = zip(vect.get_feature_names(),
                 np.asarray(dtm.sum(axis=0)).ravel())
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    sorted_scores = dict(sorted_scores)
    return sorted_scores


# <h3> Sentence Score </h3>

# In[5]:


def sentence_score(text,sorted_scores):
    sentence_score = []
    for each in text:
        score = 0.0
    #     each = each.
        for word in each.split(" "):
            #print word
            word = word.lower()
            if word in sorted_scores:
    #             print "Hello"
                score += sorted_scores[word]
    #             print word_scores[word]
        sentence_score.append([each,score])    
    return sentence_score


# <h2> TextRank </h2>
# <h3> Sentence Sim </h3>

# In[6]:


def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
 
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
 
    all_words = list(set(sent1 + sent2))
 
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
 
    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
 
    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
#     print vector1,vector2
    return 1 - cosine_distance(vector1, vector2)


# <h3> Similarity with Word2Vec </h3>

# In[7]:


def sim_scores(text):
    v1 = np.zeros(150)
    scores = []
    model = Word2Vec.load('model')
    for s in text:
        tokenizer = RegexpTokenizer(r'\w+')
        sentence = tokenizer.tokenize(s)
        for w in sentence:
            if w in model:
                v1 = v1 + model[w]
        scores.append([s,v1])
    
    from nltk.cluster.util import cosine_distance

    sim = np.zeros([len(scores),len(scores)])
    for i in xrange(len(scores)):
        for j in xrange(len(scores)):   
            sim[i][j] = cosine_distance(scores[i][1],scores[j][1])
    return sim


# <h3> Build Similarity Matrix </h3>

# In[8]:


def build_similarity_matrix(sentences, stopwords=None):
    # Create an empty similarity matrix
    S = np.zeros((len(sentences), len(sentences)))
 
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:
                continue
 
            S[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stopwords)
 
    # normalize the matrix row-wise
    for idx in range(len(S)):
        S[idx] /= S[idx].sum()
 
    return S


# <h3> PageRank </h3>

# In[9]:


def pagerank(A, eps=0.0001, d=0.85):
    P = np.ones(len(A)) / len(A)
    while True:
        new_P = np.ones(len(A)) * (1 - d) / len(A) + d * A.T.dot(P)
        delta = abs((new_P - P).sum())
        if delta <= eps:
            return new_P
        P = new_P


# <h3> TextRank </h3>

# In[10]:


def textrank(sentences,method, top_n=4, stopwords=None):
    """
    sentences = a list of sentences [[w11, w12, ...], [w21, w22, ...], ...]
    top_n = how may sentences the summary should contain
    stopwords = a list of stopwords
    """
    if method == "word_count":
        S = build_similarity_matrix(sentences, stopwords)
    elif method == "word2vec":    
        S = sim_scores(sentences)

    sentence_ranks = pagerank(S)
    # Sort the sentence ranks
    ranked_sentence_indexes = [item[0] for item in sorted(enumerate(sentence_ranks), key=lambda item: -item[1])]
    selected_sentences = sorted(ranked_sentence_indexes[:top_n])
    summary = itemgetter(*selected_sentences)(sentences)
    machine_summary = ""
    for each in summary:
        machine_summary = machine_summary+each+" "
    return machine_summary


# <h2> Calculate Rouge </h2>

# In[11]:


def rougeN(gold_summary,machine_summary,n):
    stemmer = SnowballStemmer("english")

    gold_summary_slammed = ""
    for word in gold_summary.split():
        gold_summary_slammed = gold_summary_slammed+stemmer.stem(word)+" "


    machine_summary_slammed = ""
    for word in machine_summary.split():
        machine_summary_slammed = machine_summary_slammed+stemmer.stem(word)+" "

    n_gold = ngrams(gold_summary_slammed.split(" "),n)
    n_machine = ngrams(machine_summary_slammed.split(" "),n)

    gold_list = []

    for gram in n_gold:
        gold_list.append(gram)

    machine_list = []
    for gram in n_machine:
        machine_list.append(gram)

    return float(len(list(set(gold_list).intersection(machine_list))))/float(len(list(set(gold_list))))


# <h2> Find Summary </h2>

# In[12]:


# def summary(SUMMARY_COMP_FACT,sentences):
#     scores = []
#     top_sentences = sorted(dict(sentences).items(), key=lambda x: x[1], reverse=True)[:(len(sentences)/SUMMARY_COMP_FACT)]
#     machine_summary = ""
#     for each in top_sentences:
#         machine_summary = machine_summary + each[0]
#     scores.append([SUMMARY_COMP_FACT, len(top_sentences), rougeN(gold_summary,machine_summary,2), rougeN(gold_summary,machine_summary,1)])


# In[13]:


def summarize(sentences):
    paired_sens = {}
    num_sen = len(sentences)/6

    for pair in enumerate(sentences):
         paired_sens[pair[0]]=pair[1][1]

    temp = sorted((paired_sens).items(), key=lambda x: x[1], reverse=True)[:int(num_sen)]
    temp = sorted(dict(temp).items(), key=lambda x: x[0], reverse=False)[:int(num_sen)]
    machine_summary = ""
    for i in temp:
#         print sentences[i[0]][0],"\n"
        machine_summary = machine_summary +  sentences[i[0]][0] + "\n"
    return machine_summary

def wordcount(text):
	count = 0
	for each in text:
		count = count + len(text)
	return count

# <h2> Main </h2>

# In[ ]:


from nltk.corpus import brown, stopwords


df = pd.read_csv('news_summary.csv')
result = []
f = open("results3.csv","w+")
f.write("article No. \t Words \t sentences \t Luhn Rouge1 \t Luhn Rouge2 \t TextRank-Word_count Rouge1 \t TextRank-Word_count Rouge2 \t TextRank-word2vec Rouge1 \t TextRank-word2vec Rouge2")
for i in range(3802,4300):
    print i
    f.write("\n")
    try:
        url = df['read_more'][i]
        gold_summary = df['text'][i]
        if "indiatoday" in url or "intoday" in url:
            baseurl = "i"
        elif "hindustantimes" in url:
            baseurl = "h"
        elif "theguardian" in url:
            baseurl = "t"

        file = urllib.urlopen(url)
        html = file.read()

        text = clean_data(html,baseurl)
        ws = word_scores(text)

        ss = sentence_score(text,ws)

        machine_summary_1 = summarize(ss)
        summary = []
        # TextRank with Word Count
        machine_summary_2 = textrank(text, "word_count",len(text)/5,stopwords=stopwords.words('english'))
        # TextRank with Word2Vec
        machine_summary_3 = textrank(text, "word2vec",len(text)/5,stopwords=stopwords.words('english'))
          
#         print "\n"
#         print i
#         print "Luhn's Approach    "+ "\t" +str(rougeN(gold_summary,machine_summary_1,2))+"\t"+str(rougeN(gold_summary,machine_summary_1,1))
#         print "TextRaNk - Word Count"+ "\t" + str(rougeN(gold_summary,machine_summary_2,2))+ "\t" +str(rougeN(gold_summary,machine_summary_2,1))
#         print "TextRank - Word2Vec"+ "\t" + str(rougeN(gold_summary,machine_summary_3,2))+ "\t" +str(rougeN(gold_summary,machine_summary_3,1))

        f.write(str(i)+"\t"+str(wordcount(text))+"\t" + str(len(text))+"\t" +str(rougeN(gold_summary,machine_summary_1,1))+"\t"+str(rougeN(gold_summary,machine_summary_1,2)))
        f.write("\t" + str(rougeN(gold_summary,machine_summary_2,1))+ "\t" +str(rougeN(gold_summary,machine_summary_2,2)))
        f.write("\t" + str(rougeN(gold_summary,machine_summary_3,1))+ "\t" +str(rougeN(gold_summary,machine_summary_3,2)))

#         result.append([i, rougeN(gold_summary,machine_summary,2),rougeN(gold_summary,machine_summary,1)])
    except:
        print "UnicodeDecodeError"
        
f.close()

