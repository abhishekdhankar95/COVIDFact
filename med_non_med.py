
import gensim.downloader as api
from pickle import load, dump
import numpy as np
from pandas import DataFrame, read_csv, concat
from random import shuffle
from gensim.summarization.textcleaner import split_sentences
from nltk.tokenize import word_tokenize
from sklearn import svm
import re
from sklearn.model_selection import train_test_split
import ast
from sklearn.metrics import precision_recall_fscore_support
from gensim.models import KeyedVectors
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

def conversion(training_data_csv, vector_csv):
	training_data_csv.dropna(inplace=True)
	
	for index, datapoint in training_data_csv.iterrows():
		temp_list = []
		for sentence in split_sentences(datapoint.text):
			sentence = re.sub("[^A-Za-z]+", ' ', str(sentence)).lower()
			sentence = re.sub(r'\s+', ' ', str(sentence))
			temp_list.append(encode(word_tokenize(sentence)))
		temp_df = np.array(temp_list)
		temp_df = np.average(temp_df, axis=0)
		temp_df = DataFrame([[str(temp_df), datapoint.label]], columns=['vector', 'label'])
		temp_df.to_csv(vector_csv, index=False, header=False, mode='a')


og_encoder = KeyedVectors.load_word2vec_format('word2vec_twitter_tokens_getit_actual.bin', binary=True, unicode_errors='ignore')
encoder = load(open('w2v_model_pubmed_opinion_sk_5_5.p', 'rb'))
nn_model = load(open('trained_regressor_model_pubmed_opinion_sk_5_5.p', 'rb'))

data_filename = 'tweets_health_curated_list_mod.csv'
training_data_filename = 'training_data_sick.csv'
testing_data_filename = 'testing_data_sick.csv'

vector_data_final = 'vector_data_og.csv'
vector_testdata_final = 'testset_final_sick.csv'

df = DataFrame(columns=['vector', 'label'])
df.to_csv(vector_data_final, index=False)

def encode(tokens, label=None):
	X = []
	y = []
	for token in tokens:
		if token in og_encoder.vocab:
			X.append(og_encoder[token])
			y.append(label)
	if not X:
		df = np.zeros(encoder.vector_size)
		return df
	else:
		df = np.array(X)
	return np.average(df, axis=0)

def tokenize(sentence):
	tokens = word_tokenize(sentence)
	return tokens
	
def convert_str_dataframe(df_vector, X, y):
	for index, lst_string_temp in df_vector.iterrows():
		X_ = re.sub('\n', '', lst_string_temp.vector)
		y_ = lst_string_temp.label
		X_ = re.sub('\s+', ',', X_)
		X_ = re.sub('\[\,', '[', X_)
		X.append(ast.literal_eval(X_))
		y.append(y_)
	

if __name__=='__main__':
	training_data_csv = read_csv(data_filename, usecols=['id', 'text', 'label'], dtype={'id':str, 'desc':str, 'label':str})	
	
	conversion(training_data_csv, vector_data_final)
	
	X_train = []
	y_train = []
	X_test = []
	y_test = []
	
	training_data_vector_training_shuffle = read_csv(vector_data_final)
	
	convert_str_dataframe(training_data_vector_training_shuffle, X_train, y_train)
	'''
	parameters = {'kernel':['rbf'], 'C':[10]}
	svc = svm.SVC(gamma='scale', probability=True, random_state=2)
	clf = GridSearchCV(svc, parameters, cv=5, scoring=['precision_macro', 'recall_macro', 'f1_macro'], refit='precision_macro', verbose=10)
	clf.fit(X_train, y_train)
	#print(clf.score(X_test, y_test))
	dump(clf, open('og_google.p', 'wb'))
	'''
	
	'''
	parameters = {'solver':['newton-cg', 'sag', 'saga', 'liblinear', 'lbfgs'], 'C':[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000], 'penalty':['l2']}
	lr = LogisticRegression(random_state=2)
	clf = GridSearchCV(lr, parameters, cv=5, scoring=['precision_macro', 'recall_macro', 'f1_macro'], refit='precision_macro', verbose=10)
	clf.fit(X_train, y_train)
	#print(clf.predict(X_test))
	dump(clf, open('og_google.p', 'wb'))
	'''
	
	'''
	parameters = {'criterion':['gini', 'entropy'], 'n_estimators':[10, 100, 1000, 10000], 'random_state':[2]}
	rfc = RandomForestClassifier()
	clf = GridSearchCV(rfc, parameters, cv=5, scoring=['precision_macro', 'recall_macro', 'f1_macro'], refit='precision_macro', verbose=10)
	clf.fit(X_train, y_train)
	dump(clf, open('og_google.p', 'wb'))
	'''
	
	parameters = {}
	gnb = GaussianNB()
	clf = GridSearchCV(gnb, parameters, cv=5, scoring=['precision_macro', 'recall_macro', 'f1_macro'], refit='precision_macro', verbose=10)
	clf.fit(X_train, y_train)
	dump(clf, open('og_google.p', 'wb'))
	
	'''
	clf = MLPClassifier(hidden_layer_sizes=(300, 600, 300, 50 ), random_state=1, max_iter=3000).fit(X_train, y_train)
	print(clf.loss_)
	print(clf.score(X_test, y_test))
	'''
