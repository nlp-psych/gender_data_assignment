# detect author gender from essay corpus

from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
import feature_extractor as fe
import numpy
import re
import sys
import datetime

datadir = sys.argv[1]
genderlabfile = datadir + "/BAWE_balanced_subset.csv"
conffile = sys.argv[2]

def load_balanced_gender_labels():
	'''
	Read the gender labels file and return dictionary mapping student id
	to gender
	'''
	meta_lines = [line.rstrip().split(',') for line in open(genderlabfile)]
	gender_dict = {row[0]:row[1] for row in meta_lines[1:]}
	return gender_dict

def load_essays(gender_dict):
	essays = []
	genderlabels = []
	students = []
	for student, gender in gender_dict.items():
		with open('%s/%s.txt' % (datadir, student)) as f:
			text = f.read()
			text = re.sub('<[^<]+?>', '', text)		# remove vestigial xml
			essays.append(text)
			genderlabels.append(gender)
			students.append(student)
	return essays, genderlabels, students

def load_conf_file():
	conf = set(line.strip() for line in open(conffile))
	return conf

def predict_gender(X, Y):
	scores = cross_val_score(GaussianNB(), X, Y, scoring='accuracy', cv=10)
	return scores.mean()

if __name__ == "__main__":
	gender_dict = load_balanced_gender_labels()
	essays, genderlabels, students = load_essays(gender_dict)
	conf = load_conf_file()
	features = fe.extract_features(essays, conf)

	score = predict_gender(features, genderlabels)
	print(score)

	with open('experiments.csv', 'a') as f:
		f.write(",".join(('{:%Y-%m-%d,%H:%M:%S}'.format(datetime.datetime.now()), "|".join(conf), str(score))))
