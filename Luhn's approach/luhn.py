import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from collections import Counter

# read article from file
f = open("article",'r')
text =  f.read()
f.close()

# tokenize the article
tokenizer = RegexpTokenizer(r'\w+')
word_list = tokenizer.tokenize(text)


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
# calculate sentence score
while "\n" in text:
	text = text.split("\n")

sentenceScore  = []
for sentence in text:
	score = 0
	count = 0
	for word in important_words:
		if word not in stopwords.words('english'):
			count = count +1
		if word in sentence and word not in stopwords.words('english'):
			score = score + 1
	sentenceScore.append([sentence,float(score)/float(len(important_words))**(1/2)])

# extract top n/5 sentences (n is total number of sentences in the text)
sen = sentenceScore
sentenceScore.sort(key=lambda x: x[1], reverse=True)
print len(sentenceScore)
cutScore = sentenceScore[len(sentenceScore)/7][1]

count = 0
for sentence in sen:
	if sentence[1]>cutScore:
		print sentence[0] + "\n"
		count = count + 1
print count