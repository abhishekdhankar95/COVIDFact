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

og_embedding = 'word2vec_twitter_tokens_getit_actual.bin' # should be bin
custom_embedding = 'w2v_model_pubmed_opinion_sk_5_5.p'
augmented_embedding_neural_net = 'trained_regressor_model_pubmed_opinion_sk_5_5.p'
custom_embedding_text_data = sys.argv[1]

def remove_tags(text):
    if str(text) == 'nan':
        return ''
    # remove tags newline and any url links in that order
    return sub('(<.*?>)|(\\n)|((https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,}))', '', text)


word_vectors = KeyedVectors.load_word2vec_format(og_embedding, binary=True, unicode_errors='ignore')
print('upload word2vec_twitter_tokens_getit_actual.bin')

nlp = spacy.load('en') # disabling Named Entity Recognition for speed
stop_words = set(['were','be','been','being','doing','a','an','the','and','but','if','or','because','as','until','while','of','at','by','for','with','about','between','into','to','from','up','down','in','out','on','off','over','under','again','this','further','then','once','here','there','such','no','nor','not','only','own','so','than','too','very','s','t','now','help','please','pls'])
def cleaning(doc):
    txt = [token.text for token in doc if token.text not in stop_words]
    # only return text if contains more than 2 words
    if len(txt) > 2:
        return ' '.join(txt)

print("written header to temporary clean file")
DataFrame(columns=['temp_clean']).to_csv('temporary_clean_text.csv', index=False, mode='w+')

print("read unprocessed text")
df = read_csv(custom_embedding_text_data, encoding='utf-8')

print('replace all nan with empty string')
df.replace(nan, '', regex=True, inplace=True)
print("drop all nan")
df = df.dropna().reset_index(drop=True)
df = DataFrame(df.text.unique(), columns=['text'])

print("remove tags from unprocessed text and write to temporary clean text")

for i in range(len(df)):
    
    DataFrame(split_sentences(remove_tags(df.iloc[i]['text'])), columns=['temp_clean'], dtype=str).\
                                                    to_csv('temporary_clean_text.csv', header=False, index=False, mode='a')


logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)

del df
gc.collect()

print("read temporary clean text to convert to lower case")
df_temp_clean = read_csv('temporary_clean_text.csv', usecols=['temp_clean'], dtype={'temp_clean':str}, lineterminator='\n')
brief_cleaning = (sub("[^A-Za-z]+", ' ', str(row)).lower() for row in df_temp_clean['temp_clean'])

t = time()
txt = []

del df_temp_clean
gc.collect()

print("write header for final_clean_text.csv")
DataFrame(columns=['final_clean']).to_csv('final_clean_text.csv', index=False, mode='w+')
print(locals())
print("cleaning temporary clean text nlp.pipe and writing to final clean text")
for doc in nlp.pipe(brief_cleaning, batch_size=10000, n_threads=-1):
    DataFrame([cleaning(doc)], columns=['final_clean'], dtype=str) \
                                                                        .to_csv('final_clean_text.csv', index=False, header=False, mode='a+')
    #print('10000 done')

#print('Time to clean up everything: {} mins'.format(round((time() - t) / 60, 2)))
df_clean = read_csv('final_clean_text.csv', usecols=['final_clean'], dtype={'final_clean':str})
df_clean = df_clean.dropna().drop_duplicates()
sentences = [row.split() for row in df_clean['final_clean']]
del df_clean
gc.collect()


cores = multiprocessing.cpu_count() # Count the number of cores in a computer
w2v_model = Word2Vec(min_count=5,
                     window=5,
                     size=400,
                     sample=1e-5, 
                     #alpha=0.03, 
                     #min_alpha=0.0007, 
                     negative=3,
                     workers=cores,
                     sg=1,
                     hs=0)
                     
t = time()

w2v_model.build_vocab(sentences, progress_per=10000)

print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))
t = time()

w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=50, report_delay=1)

print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))
w2v_model.init_sims(replace=True)

pickle.dump(w2v_model, open(custom_embedding, "wb"))



#w2v_model = pickle.load(open(custom_embedding, 'rb'))

#word_vectors = api.load("word2vec-google-news-300")
og_vocab = set(word_vectors.vocab)
ndd_vocab = set(w2v_model.wv.vocab)
og_ndd_intersection_vocab = og_vocab.intersection(ndd_vocab)
mlp_regressor = nn.MLPRegressor(hidden_layer_sizes=(word_vectors.vector_size,), solver='sgd', max_iter=5000)
#mlp_regressor.batch_size = 1
#X = zeros(word_vectors.vector_size)
X = []
#y = zeros(word_vectors.vector_size)
y = []
for i in og_ndd_intersection_vocab:
    #X = vstack((X, word_vectors.get_vector(i)))
    X.append(word_vectors.get_vector(i))
    #y = vstack((y, w2v_model.wv.get_vector(i)))
    y.append(w2v_model.wv.get_vector(i))


X = array(X)
y = array(y)

trained_regressor_model = mlp_regressor.fit(X, y)

pickle.dump(trained_regressor_model, open(augmented_embedding_neural_net, "wb"))
