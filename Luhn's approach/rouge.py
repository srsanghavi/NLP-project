from nltk import ngrams
from collections import Counter
from nltk.stem.snowball import SnowballStemmer

n = 1

gold_summary = "The Administration of Union Territory Daman and Diu has revoked its order that made it compulsory for women to tie rakhis to their male colleagues on the occasion of Rakshabandhan on August 7. The administration was forced to withdraw the decision within 24 hours of issuing the circular after it received flak from employees and was slammed on social media."
# machine_summary = "In this connection all offices departments shall remain open and celebrate the festival collectively at a suitable time wherein all the lady staff shall tie rakhis to their colleagues, the order, issued on August 1 by Gurpreet Singh, deputy secretary (personnel), had said. In 2014, the year BJP stormed to power at the Centre, Rashtriya Swayamsevak Sangh (RSS) chief Mohan Bhagwat said the festival had national significance and should be celebrated widely 'to protect Hindu culture and live by the values enshrined in it' he Daman and Diu administration on Wednesday withdrew a circular that asked women staff to tie rakhis on male colleagues after the order triggered a backlash from employees and was ripped apart on social media. Rakshabandhan, a celebration of the bond between brothers and sisters, is one of several Hindu festivities and rituals that are no longer confined of private, family affairs but have become tools to push politic al ideologies"
machine_summary = "A notification issued by the Daman and Diu administration made it compulsory for women to tie rakhis to their male colleagues on the occasion of Rakshabandhan. The two notifications  one mandating the celebration of Rakshabandhan (left) and the other withdrawing the mandate (right)  were issued by the Daman and Diu administration a day apart. The circular was withdrawn through a one-line order issued late in the evening by the UTs department of personnel and administrative reforms. The notice was issued on Daman and Diu administrator and former Gujarat home minister Praful Kodabhai Patels direction, sources said."

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

print float(len(list(set(gold_list).intersection(machine_list))))/float(len(list(set(gold_list))))