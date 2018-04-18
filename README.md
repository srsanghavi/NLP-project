# News Article Summarization
Project for CS 6120 Natural Language Processing, Sp 2018<br>
Northeastern University<br>
Language for implementation: python (Jupyter notebook)<br>

Dependencies used: NLTK, gensim, urlib, numpy, beautifulsoup, scikit-learn, pandas, matplotlib<br>

This repository contains a folder "implementation", project report and a presentation.<br>
	- Presentation : brief overview about the work<br>
	- Report : Detailed description of the work<br>
	- implementation : Python scripts to run the experiments<br>

Implementation folder contains:<br>
<br>
1) news_summary.csv<br>
	- is a dataset with around 4300 links to webpagegs, their human summaries and other article details (obtained from kaggle)<br>

2) extraction_modular.ipynb:<br>
	- performs text summarization with Luhn's approach using TF-IDF and Stemming, TextRank and TextRank with Word2Vec<br>
	- Required : Word2Vec model file, dataset<br>
	-Output: a csv file containning Rouge-1 and Rouge-2 score for each link in the dataset<br>
3) Analyze results.ipynb:<br>
	- contains the implementation to analyze, generate and compare the Rouge-N scores of summaries.<br>
4) doc2vec.ipynb:<br>
	-  generates (trains) the word2vec model from 100 articles which is used in TextRank algorithm to generate sentence vectors.<br>
	

To run the files: <br>
	1) run following command from the project directory<br>
		- $jupyter notebook<br>
	2) it will open a notebook in browser<br>
	3) open code in the browser from the list<br>
	4) press shift + Enter to run each cell in the notebook<br>
	5) ouput for each cell would be printed in the same notebook<br>
	
References for the implementation:<br>
1] http://nlpforhackers.io/textrank-text-summarization/<br>
2] https://radimrehurek.com/gensim/models/word2vec.html<br>
3] http://scikit-learn.org/stable/modules/feature_extraction.html<br>
