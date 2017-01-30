# Data assignment \#1: Gender classification from text

## Natural Language Processing and Psychology (Corpus Analysis)

### Data

The data for this assignment is a subset of the British National Corpus: 570 essays, 285 written by female authors, 285 by male authors.

The full set of essays is in CORPUS\_TXT. The essays to be used in this assignment are listed in CORPUS\_TXT/BAWE\_balanced\_subset.csv. The “fname” column is the essay filename (CORPUS\_TXT/[fname].csv) and the “gender” column is the author gender (0=male, 1=female). Disclosure: We selected a subset of essays that are “easier” to classify.

### Task

You are provided with two scripts: bawe\_gender\_classifier.py and feature\_extractor.py. 

bawe\_gender\_classifier.py loads the essays specified in BAWE\_balanced\_subset.csv, calls extract\_features from feature\_extractor to extract a set of features, and runs a NaiveBayes classifier to predict the gender labels from the features. You don't need to do anything to this script.

feature\_extractor.py is a collection of functions to extract various features from text data, which it expects to have the form of a list of strings, where each string is an instance to be classified. Here, you will write functions to extract a subset of the following features (based on the Argamon and Ottenbacher papers):

* Function words: counts for each function word (from the nltk stopwords list). __This first feature set is implemented for you.__

* POS: counts for 500 most common ordered POS tag triples, 100 most common ordered POS tag pairs, all POS single tags.

* Lexical: counts for 500 most common trigrams, 100 most common bigrams, 100 most common unigrams (stopwords removed for unigram counts)

* __Note: divide each count by the total number of words in the essay.__

* Topic: score for 20 highest topic models.

* Complexity: average number of characters per word, #unique words/# total words, average #words per sentence.

* __Preprocessing__: before extracting features, stem all words and replace any that appear only once with the token <RARE>. __This preprocessing step has been implemented for you.__

Then, using the bawe\_gender\_classifier.py script, get the gender detection accuracy using each individual feature set, as well as all the features combined. Fill out the submission.md worksheet.

See <http://www.nyu.edu/projects/politicsdatalab/localdata/workshops/NLTK_presentation%20_code.py> for excellent focused nltk tutorial.

### Github workflow for submission (beta)

We will use the [fork and pull request](https://guides.github.com/activities/forking/)  workflow for submissions for this class. Follow the link for a guide to this workflow. The idea is that you will fork this repository, make the required changes in your own copy, and then submit a pull request and @mention me (@rivlev). __The pull request and @mention comprise your submission for this assignment.__ I will look at your feature\_extractor.py and submission.md.

This setup for this class is new, so please submit a separate pull request if you can clear up any errors or lack of clarity here. 
