
# coding: utf-8

# In[1]:


import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.corpus import brown
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np

NOUNS = ['NN', 'NNS', 'NNP', 'NNPS']
# In[2]:


# read article from file
f = open("article",'r')
text =  f.read().decode('unicode_escape').encode('ascii','ignore')
f.close()

# tokenize the article
tokenizer = RegexpTokenizer(r'\w+')
word_list = tokenizer.tokenize(text)


# remove stop words from the article
filtered_words = [word for word in word_list if word.lower() not in stopwords.words('english')]
total_words = len(filtered_words)
words = filtered_words

# print words
# In[3]:


# # find n-gram probability for filtered words
# filtered_words  = []
# for each in Counter(words).items():
#     filtered_words.append([each[0],float(each[1])/float(total_words)])


# In[4]:


train_filtered_words = []
# brown corpus traning vectorizer
train_data = brown.words(categories='news')
# remove stop words from the article
train_filtered_words = [word for word in train_data if word.lower() not in stopwords.words('english')]
total_words = len(train_filtered_words)
train_words = train_filtered_words
# print train_words

# In[ ]:


count_vect = CountVectorizer()
count_vect = count_vect.fit(train_words)
freq_term_matrix = count_vect.transform(train_words)
feature_names = count_vect.get_feature_names()


# In[ ]:


tfidf = TfidfTransformer(norm="l2")
tfidf.fit(freq_term_matrix)


# In[ ]:


doc_freq_term = count_vect.transform(words)
doc_tfidf_matrix = tfidf.transform(doc_freq_term)
story_dense = doc_tfidf_matrix.todense()
doc_matrix = story_dense.tolist()[0]


# In[ ]:


def similarity_score(t, s):
#     t = remove_stop_words(t.lower())
#     s = remove_stop_words(s.lower())
    t_tokens, s_tokens = t, s
    similar = [w for w in s_tokens if w in t_tokens]
    score = (len(similar) * 0.1 ) / len(t_tokens)
    return score


# In[ ]:


title = "Rakshabandhan compulsory for employees in Daman and Diu"

# tokenize the article
tokenizer = RegexpTokenizer(r'\w+')
title_list = tokenizer.tokenize(title)


# remove stop words from the article
title_words = [word for word in title_list if word.lower() not in stopwords.words('english')]
total_words = len(title_words)
t_words = title_words

def rank_sentences(doc, doc_matrix, feature_names, top_n=3):
    sents = nltk.sent_tokenize(doc)
    sents = doc
    sentences = [nltk.word_tokenize(sent) for sent in sents]
    sentences = [[w for w in sent if nltk.pos_tag([w])[0][1] in NOUNS]
                  for sent in sentences]
    tfidf_sent = [[doc_matrix[feature_names.index(w.lower())]
                   for w in sent if w.lower() in feature_names]
                 for sent in sentences]

    # Calculate Sentence Values
    doc_val = sum(doc_matrix)+1
    sent_values = [sum(sent) / doc_val for sent in tfidf_sent]

    # Apply Similariy Score Weightings
    similarity_scores = [similarity_score(t_words, sent) for sent in sents]
    scored_sents = np.array(sent_values) + np.array(similarity_scores)

    # Apply Position Weights
    ranked_sents = [sent*(i/len(sent_values))
                    for i, sent in enumerate(sent_values)]

    ranked_sents = [pair for pair in zip(range(len(sent_values)), sent_values)]
    ranked_sents = sorted(ranked_sents, key=lambda x: x[1] *-1)
    print ranked_sents
    return ranked_sents[:top_n]


# In[ ]:


top_sents = rank_sentences(text, doc_matrix, feature_names)

# print top_sents
# In[ ]:


# top_sents = rank_sentences(doc, doc_matrix, feature_names)

