# extract features from list of text instances based on configuration set of features

import nltk
import numpy
import re
import csv

source_text = []
stemmed_text = []

def preprocess():
	# first stem and lowercase words, then remove rare
	# lowercase 
	global source_text
	source_text = [text.lower() for text in source_text]

	# tokenize
	tokenized_text = [nltk.word_tokenize(text) for text in source_text]
	
	# stem
	porter = nltk.PorterStemmer()
	global stemmed_text
	stemmed_text = [[porter.stem(t) for t in tokens] for tokens in tokenized_text]

	# remove rare
	vocab = nltk.FreqDist(w for line in stemmed_text for w in line)
	rarewords_list = set(vocab.hapaxes())
	stemmed_text = [['<RARE>' if w in rarewords_list else w for w in line] for line in stemmed_text]
	# note that source_text will be lowercased, but only stemmed_text will have rare words removed

def bag_of_function_words():
	bow = []
	for sw in nltk.corpus.stopwords.words('english'):
		counts = [sum(1 for _ in re.finditer(r'\b%s\b' % sw, line)) for line in source_text]
		bow.append(counts)
	return bow, nltk.corpus.stopwords.words('english')

# FILL IN OTHER FEATURE EXTRACTORS

def log(fvec, hvec):
	with open('log.csv', 'a') as lfile:
		lwriter = csv.writer(lfile)
		lwriter.writerow(hvec)
		lwriter.writerows(fvec)

def extract_features(text, conf):
	all = False
	if len(conf)==0:
		all = True

	global source_text
	source_text = text			# we'll use global variables to pass the data around
	preprocess()

	features = []		# features will be list of lists, each component list will have the same length as the list of input text
	header = []


	# extract requested features: FILL IN HERE
	if 'bag_of_function_words' in conf or all:
		fvec, hvec = bag_of_function_words()
		features.extend(fvec)
		header.extend(hvec)
		log(fvec, hvec)

	features = numpy.asarray(features).T.tolist() # transpose list of lists sow its dimensions are #instances x #features

	with open('features.csv', 'w') as ffile:
		fwriter = csv.writer(ffile)
		fwriter.writerow(header)
		fwriter.writerows(features)

	return features
