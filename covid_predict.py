
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

def conversion(training_data_csv, vector_csv):
	training_data_csv.dropna(inplace=True)
	# brief_cleaning = (sub("[^A-Za-z]+", ' ', str(row)).lower() for row in df_temp_clean['temp_clean'])
	
	for index, datapoint in training_data_csv.iterrows():
		temp_list = []
		for sentence in split_sentences(datapoint.text):
			sentence = re.sub("[^A-Za-z]+", ' ', str(sentence)).lower()
			sentence = re.sub(r'\s+', ' ', str(sentence))
			#print(word_tokenize(sentence))
			temp_list.append(encode(word_tokenize(sentence)))
		temp_df = np.array(temp_list)
		temp_df = np.average(temp_df, axis=0)
		temp_df = DataFrame([[str(temp_df)]], columns=['vector'])
		temp_df.to_csv(vector_csv, index=False, header=False, mode='a')


# google only modification 
# google only csvs ok

#vector_csv = 'training_data_vector_new_combined_10.csv'

og_encoder = KeyedVectors.load_word2vec_format('E:/embeddings_twitter_pet/word2vec_twitter_tokens_getit_actual.bin', binary=True, unicode_errors='ignore')

classifier_pickle = 'covid_classifier_rbf_87.p'
predict_text_filename = 'updated_test.csv' # this is the filw which should have text csv to be classified
vector_data = 'temp_text_vector.csv'
prediction_filename = 'prediction.csv'

'''
df = read_csv(data_filename)
rs = ShuffleSplit(n_splits=1, test_size=0.20, random_state=0)
train_indices = []
test_indices = []
for train_index, test_index in rs.split(list(df.index)):
	train_indices = train_index
	test_indices = test_index

del df
#df.iloc[train_indices].to_csv(training_data_filename, columns=['id', 'text', 'label'], index=False)
#df.iloc[test_indices].to_csv(testing_data_filename, columns=['id', 'text', 'label'], index=False)
'''

# remove this when done
df_temp = DataFrame(columns=['vector'])
df_temp.to_csv(vector_data, index=False)
del df_temp

def encode(tokens, label=None):
	X = []
	y = []
	for token in tokens:
		if token in og_encoder.vocab:
			X.append(og_encoder[token])
			y.append(label)
		
		#if token in og_encoder.vocab:
		#	X.append(nn_model.predict(og_encoder[token].reshape(1, -1))[0])
		#	y.append(label)
		
		#elif token in encoder.wv.vocab:
		#	X.append(encoder.wv.get_vector(token))
		#	y.append(label)
		
		#else:
		#	#print(encoder.vector_size)
		#	X.append(np.zeros(encoder.vector_size))
		#	y.append(-1)
	if not X:
		df = np.zeros(encoder.vector_size)
		return df
	else:
		df = np.array(X)
	return np.average(df, axis=0)
	# return (X,y)


def tokenize(sentence):
	tokens = word_tokenize(sentence)
	# tokens = [t1.lower() for t1 in tokens if t1 not in punctuation] # Remove punctuations & stop words using Glasgow stop words list
	return tokens
	
def convert_str_dataframe(df_vector, X):
	for index, lst_string_temp in df_vector.iterrows():
		X_ = re.sub('\n', '', lst_string_temp.vector)
		X_ = re.sub('\s+', ',', X_)
		X_ = re.sub('\[\,', '[', X_)
		X.append(ast.literal_eval(X_))
		
def predict_text():
	covid_classifier = load(open(classifier_pickle, 'rb'))
	predict_text = read_csv(predict_text_filename, usecols=['ID', 'text'], dtype={'ID':str, 'text':str})
	conversion(predict_text, vector_data)
	X_test = []
	y_test = []
	convert_str_dataframe(read_csv(vector_data), X_test)
	y_test = covid_classifier.predict(X_test)
	print(len(y_test), len(X_test))
	print(y_test)
	concat([predict_text.text, \
	DataFrame(y_test, columns=['prediction']).prediction], axis=1)\
	.to_csv(prediction_filename, index=False, header=['text', 'prediction'])
	

if __name__=='__main__':
	predict_text()
	





