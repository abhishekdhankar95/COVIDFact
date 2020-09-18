from gensim.models.phrases import Phrases, Phraser
import pickle
from nltk import word_tokenize
from nltk.util import ngrams
from nltk import download
from nltk.lm import NgramCounter
from pandas import read_csv
from pickle import dump, load

treatment_word_list = ['treatment', 'treated', 'treats', 'treat', 'treating', 'therapy', 'therapeutic']
cause_word_list = ['aethiology', 'aetiology', 'etiology', 'causes', 'cause', 'caused', 'causing', 'aitiology', 'causality', 'causal', 'causation', 'causativity', 'causative', 'causable', 'causability', 'induce']
prevent_word_list = ['preventable', 'prevent', 'prevents', 'preventing', 'prevented', 'preventability', 'preventible', 'preventiveness', 'preventive', 'prevention', 'prophylaxis', 'prophylax', 'prophylactic', 'prophylaxis']



if __name__=='__main__':
	cause_tweets = []
	med_text = read_csv('grebe_covid_med_for_hamman_with_vector.csv', usecols=['text'], dtype=str)
	for text in med_text.text:
		if set(cause_word_list).intersection(set(text.split())):
			cause_tweets.append(1)
		else:
			cause_tweets.append(0)
	dump(cause_tweets, open('cause_tweets.p', 'wb'))
	
	
	
	
	
