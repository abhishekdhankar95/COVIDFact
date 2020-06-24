from keras.preprocessing.text import Tokenizer
from time import time  # To time operations
from collections import defaultdict  # For word frequency
from gensim.models.phrases import Phrases, Phraser
from gensim.summarization.textcleaner import clean_text_by_sentences, split_sentences
import multiprocessing
import gensim.downloader as api
import sklearn.neural_network as nn
from gensim.models import Word2Vec
import spacy  # For preprocessing
import logging  # Setting up the loggings to monitor gensim
from gensim.test.utils import get_tmpfile, datapath
from gensim.models import KeyedVectors
import sklearn.neural_network as nn
import pickle
import sys
from pandas import read_csv, DataFrame
from numpy import nan, str, zeros, vstack, array
from re import sub
import gc
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from keras.layers import Embedding
from keras.metrics import Precision, Recall, categorical_accuracy
from keras import Sequential
from keras.utils import to_categorical
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Dense, Dropout, Flatten
from re import sub

# Prepare Glove File
def readGloveFile(gloveFile):
    
    wordToVec = {}  # map from a token (word) to a Glove embedding vector
    wordToIndex = {}  # map from a token to an index
    indexToWord = {}  # map from an index to a token 

    wordToVec = KeyedVectors.load_word2vec_format('word2vec_twitter_tokens_getit_actual.bin', binary=True, unicode_errors='ignore').wv

    tokens = sorted(wordToVec.keys())
    for idx, tok in enumerate(tokens):
        kerasIdx = idx + 1  # 0 is reserved for masking in Keras (see above)
        wordToIndex[tok] = kerasIdx # associate an index to a token (word)
        indexToWord[kerasIdx] = tok # associate a word to a token (word). Note: inverse of dictionary above

    return wordToIndex, indexToWord, wordToVec

# Create Pretrained Keras Embedding Layer
def createPretrainedEmbeddingLayer(wordToVec, wordToIndex, isTrainable):
    vocabLen = len(wordToIndex) + 1  # adding 1 to account for masking
    embDim = next(iter(wordToVec.values())).shape[0]  # works with any glove dimensions (e.g. 50)

    embeddingMatrix = np.zeros((vocabLen, embDim))  # initialize with zeros
    for word, index in wordToIndex.items():
        embeddingMatrix[index, :] = wordToVec[word] # create embedding: word index to Glove word embedding

    embeddingLayer = Embedding(vocabLen, embDim, weights=[embeddingMatrix], trainable=isTrainable)
    return embeddingLayer
'''
# usage
wordToIndex, indexToWord, wordToVec = readGloveFile("/path/to/glove.6B.50d.txt")
pretrainedEmbeddingLayer = createPretrainedEmbeddingLayer(wordToVec, wordToIndex, False)
model = Sequential()
model.add(pretrainedEmbeddingLayer)
'''


MAX_NB_WORDS = 2000
MAX_SEQUENCE_LENGTH = 30
hidden_size = 400
number_of_classes = 3


og_embedding = 'medical_datasets/word2vec_twitter_tokens_getit_actual.bin' # should be bin
custom_embedding = 'w2v_model_pubmed_opinion_sk_5_5.p'
augmented_embedding_neural_net = 'trained_regressor_model_pubmed_opinion_sk_5_5.p'

treatment_file_name = 'medical_datasets/treats_actual_unique.csv'
cause_file_name = 'medical_datasets/causes_actual_unique.csv'
prevent_file_name = 'medical_datasets/prevents_actual_unique.csv'
medical_file_name = 'medical_datasets/actual_unique.csv'

def clean_text(text_dataframe):
    return sub('[^A-Za-z\s]', '', text_dataframe.text.str)

def get_data_text_and_label(data_csv_file_name, texts, labels, label=None):
    df = read_csv(data_csv_file_name, usecols=['text', 'label'])
    texts.extend(list(df.text))
    labels.extend(list(df.label+1))

texts = []
labels = []
'''
get_data_text_and_label(treatment_file_name, texts, labels, -1)
get_data_text_and_label(cause_file_name, texts, labels, 0)
get_data_text_and_label(prevent_file_name, texts, labels, 1)
'''

get_data_text_and_label(medical_file_name, texts, labels)

'''
shuffle(text)
shuffle(labels)
'''

#twitter_embedding = KeyedVectors.load_word2vec_format('word2vec_twitter_tokens_getit_actual.bin', binary=True, unicode_errors='ignore')
#twitter_embedding = api.load("word2vec-google-news-300")
embeddings_index = KeyedVectors.load_word2vec_format(og_embedding, binary=True, unicode_errors='ignore')

EMBEDDING_DIM = embeddings_index.vector_size

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index # number of unique words (vocab)
print(list(set(labels)))
labels = to_categorical(np.asarray(labels)) # convert into one hot vectors
print(labels.shape)
print(labels[0])
#exit()
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

#X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, train_size=0.80, random_state=0)
sss.get_n_splits(data, labels)
for train_index, test_index in sss.split(data, labels):
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = None
    if word in embeddings_index:
        embedding_vector = embeddings_index[word]
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

pickle.dump(embedding_matrix, open('embedding_matrix.p', 'wb'))

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
#print(y_test.shape)
model = Sequential()
model.add(embedding_layer)
model.add(LSTM(hidden_size, return_sequences=False))
#model.add(LSTM(hidden_size, return_sequences=False))
model.add(Dropout(0.5))
#model.add(Flatten())
model.add(Dense(number_of_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])


model.fit(X_train, y_train,
        epochs=2, batch_size=20000)

