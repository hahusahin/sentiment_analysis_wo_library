# IMPORT LIBRARIES
import numpy as np 
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from TurkishStemmer import TurkishStemmer
from sklearn.metrics import classification_report

# LOAD DATASET
df = pd.read_excel("reviews.xlsx")

X = list(df['YORUM'])
y = list(df['DUYGU'].map({'olumsuz': 0, 'cekimser': 1, 'olumlu': 2}))

X_train = X[0:60]  + X[70:90] + X[95:160]
X_test  = X[60:70] + X[90:95] + X[160:175]
y_train = y[0:60]  + y[70:90] + y[95:160]
y_test  = y[60:70] + y[90:95] + y[160:175]

# HELPER FUNCTIONS
def preprocess_review(review):
	'''
	Takes a single review as an input
	Returns a processed, clean (tokenized, puntuations/stop words removed and stemmed) sentence
	'''
	tokenized = word_tokenize(review.lower(), language='turkish')
	no_punc = [t for t in tokenized if t.isalpha()]
	no_stop = [t for t in no_punc if t not in stopwords.words('turkish')]

	stemmer = TurkishStemmer()
	review_cleaned = [stemmer.stem(t) for t in no_stop]

	return review_cleaned


def frequencies(reviews, labels):
	'''
	Takes list of reviews and sentiment classes as input
	Returns a dictionary that maps each pair of (word, label) to its frequency
	'''
	freqs = {}

	for y, review in zip(labels, reviews):
		for word in preprocess_review(review):
			pair = (word, y)
			if pair in freqs:
				freqs[pair] += 1
			else:
				freqs[pair] = 1

	return freqs


def log_probs(freqs, labels):
	'''
	Takes frequencies and sentiment class labels as input
	Returns log-likelihood of each word for each class
	'''
	log_likelihoods = {}

	y_neg = labels.count(0)
	y_notr = labels.count(1)
	y_pos = labels.count(2)

	logprior_neg = np.log(y_neg / len(labels))
	logprior_notr = np.log(y_notr / len(labels))
	logprior_pos = np.log(y_pos / len(labels))
	log_priors = (logprior_neg, logprior_notr, logprior_pos)

	vocab = set([pair[0] for pair in freqs.keys()]) 

	N_neg = N_notr = N_pos = 0
	for pair in freqs.keys():
		if pair[1] == 0:
			N_neg += freqs[pair]
		elif pair[1] == 1:
			N_notr += freqs[pair]
		else:
			N_pos += freqs[pair]

	for word in vocab:
		freq_neg = freqs.get((word, 0), 0)
		freq_notr = freqs.get((word, 1), 0)
		freq_pos = freqs.get((word, 2), 0)

		p_neg = (freq_neg + 1) / (N_neg + len(vocab))
		p_notr = (freq_notr + 1) / (N_notr + len(vocab))
		p_pos = (freq_pos + 1) / (N_pos + len(vocab))

		log_likelihoods[(word, "neg")] = np.log(p_neg)
		log_likelihoods[(word, "notr")] = np.log(p_notr)
		log_likelihoods[(word, "pos")] = np.log(p_pos)
	    
	return log_priors, log_likelihoods


def predict_reviews(reviews, y_true, logpriors, log_likelihoods):
	"""
	Takes list of reviews, true labels, log-probabilities as input
	Prints the evaluation metric results
	"""
	y_preds = []

	for review in reviews:
		words = preprocess_review(review)

		p_neg = logpriors[0]
		p_notr = logpriors[1]
		p_pos = logpriors[2]

		for word in words:
			p_neg += log_likelihoods.get((word, "neg"), 0)
			p_notr += log_likelihoods.get((word, "notr"), 0)
			p_pos += log_likelihoods.get((word, "pos"), 0)

		y_preds.append(np.argmax(np.array([p_neg, p_notr, p_pos])))
	
	print()
	print(classification_report(np.array(y_true), 
	                        	np.array(y_preds), 
	                        	target_names=["olumsuz","cekimser","olumlu"]))


# MODEL
freqs = frequencies(X_train, y_train)

log_priors, log_likelihoods = log_probs(freqs, y_train)

predict_reviews(X_test, y_test, log_priors, log_likelihoods)