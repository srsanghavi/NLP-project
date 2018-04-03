
# coding: utf-8

# In[34]:


import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.corpus import brown
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


# In[42]:


# read article from file
words = []
f = open("article",'r')
text =  f.read().decode('unicode_escape').encode('ascii','ignore')
split_text = text.split("\n")
f.close()


# tokenize the article
tokenizer = RegexpTokenizer(r'\w+')
word_list = tokenizer.tokenize(text)

# remove stop words from the article
filtered_words = [word for word in word_list if word.lower() not in stopwords.words('english')]
total_words = len(filtered_words)
words = filtered_words
#print words
# print split_text


# In[43]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd

train_filtered_words = []
# # brown corpus traning vectorizer
train_data = brown.words(categories='news')
# # remove stop words from the article
train_filtered_words = [word for word in train_data if word.lower() not in stopwords.words('english')]
total_words = len(train_filtered_words)
train_words = train_filtered_words

#pd.DataFrame(vect.fit_transform(train_words).toarray(), columns=vect.get_feature_names())

# # TfidfVectorizer
vect = CountVectorizer()
#d.DataFrame(vect.fit_transform(train_words).toarray(), columns=vect.get_feature_names())
cv_words = vect.fit_transform(train_words)
print cv_words.vocabulary_.get(u'algorithm')
raw_input()

# In[59]:



# vect = TfidfVectorizer(stop_words='english')
dtm = vect.fit_transform(words)
features = vect.get_feature_names()
# print features
dtm.shape


# In[58]:


import numpy as np
# choose a random review that is at least 300 characters

review_id = np.random.randint(0, len(words))
review_text = words
review_length = len(words)

# create a dictionary of words and their TF-IDF scores
word_scores = {}
for word in words:
#     word = word.lower()
    if word in features:
        word_scores[word] = dtm[review_id, features.index(word)]

print word_scores
# print words with the top 5 TF-IDF scores
print 'TOP SCORING WORDS:'
top_scores = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:10]
for word, score in top_scores:
    print word

# print 5 random words
# print '\n' + 'RANDOM WORDS:'
random_words = np.random.choice(word_scores.keys(), size=5, replace=False)
for word in random_words:
    print word

# print the review
# print '\n' + review_text

