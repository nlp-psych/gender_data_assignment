# Data assignment \#2: Sarcasm classification from text

## Natural Language Processing and Psychology (Corpus Analysis)

### Data

The data for this assignment is a subset of the [Internet Argument Corpus](https://nlds.soe.ucsc.edu/iac), a corpus for researching political debate on internet forums. This part includes quote-reponse pairs annotated for sarcasm. The sarcasm annotations relate only to the *response*; the quote text is included for context.

The data is organized into three categories of sarcasm: general, hyperbole, and rhetorical questions. This assignment will focus on the *general* category, which has 1,630 quote-response pairs per class (sarcastic and non-sarcastic).

More information about the corpus is available [here](https://nlds.soe.ucsc.edu/sarcasm2) and [here](http://www.sigdial.org/workshops/conference17/proceedings/pdf/SIGDIAL04.pdf).

## Task

The csv file containing the sarcasm data is in data/sarcasm_v2.csv. The paper describing the data is there as well.

sarcasm_classifier.py takes a configuration file as an argument and will load the data, call the feature extractor and run an SVM classifier.

Copy over your feature extractor from the gender assignment and add the following features:

* Averaged word vectors (use gensim's word2vec implementation and Google's pre-trained vectors) ([tutorial here])(https://rare-technologies.com/word2vec-tutorial/)

* Another interesting feature (up to you)

* Experiments: 

   * Just averaged word vectors
   * Just your feature
   * Word vectors and your feature
   * Lexical and pos ngrams you used for gender
   * All features

## Dependencies

* [Gensim](https://radimrehurek.com/gensim/install.html) (you can also choose another package such as tensorflow)
* [Google News pretrained word vectors](https://docs.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download)
