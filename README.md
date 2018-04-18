# News Article Summarization
Project for CS 6120 Natural Language Processing, Sp 2018
Northeastern University
Language for implementation: python (Jupyter notebook)

Dependencies used: NLTK, gensim, urlib, numpy, beautifulsoup, scikit-learn, pandas, matplotlib

This repository contains a folder "implementation", project report and a presentation.
	- Presentation : brief overview about the work
	- Report : Detailed description of the work
	- implementation : Python scripts to run the experiments

Implementation folder contains:

1) news_summary.csv
	- is a dataset with around 4300 links to webpagegs, their human summaries and other article details (obtained from kaggle)

2) extraction_modular.ipynb:
	- performs text summarization with Luhn's approach using TF-IDF and Stemming, TextRank and TextRank with Word2Vec
	- Required : Word2Vec model file, dataset
	-Output: a csv file containning Rouge-1 and Rouge-2 score for each link in the dataset
3) Analyze results.ipynb:
	- contains the implementation to analyze, generate and compare the Rouge-N scores of summaries.
4) doc2vec.ipynb:
	-  generates (trains) the word2vec model from 100 articles which is used in TextRank algorithm to generate sentence vectors.
	

To run the files: 
	1) run following command from the project directory
		- $jupyter notebook
	2) it will open a notebook in browser
	3) open code in the browser from the list
	4) press shift + Enter to run each cell in the notebook
	5) ouput for each cell would be printed in the same notebook
	
References for the implementation:
1] http://nlpforhackers.io/textrank-text-summarization/
2] https://radimrehurek.com/gensim/models/word2vec.html
3] http://scikit-learn.org/stable/modules/feature_extraction.html
