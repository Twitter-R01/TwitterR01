import pandas as pd
import numpy as np
import pickle
import multiprocessing
import tensorflow as tf
import re
from nltk import tokenize
from bs4 import BeautifulSoup
from gensim.models import KeyedVectors
from sklearn import utils
from sklearn.model_selection import train_test_split, GridSearchCV
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.layers.embeddings import Embedding
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.callbacks import Callback, EarlyStopping

def create_lstm(max_features, embed_size, max_len, embedding_matrix, optimizer='rmsprop'):
	inputs = Input(name='inputs', shape=(max_len,))
	layer = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inputs)
	layer = LSTM(64)(layer)
	layer = Dense(256,name='FC1')(layer)
	layer = Activation('relu')(layer)
	layer = Dropout(0.5)(layer)
	layer = Dense(1,name='out_layer')(layer)
	layer = Activation('sigmoid')(layer)
	model = Model(inputs=inputs,outputs=layer)
	model.compile(loss='binary_crossentropy', optimizer=optimizer)
	return model

def create_embedding_matrix(filepath, word_index, embedding_dim):
	vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
	embedding_matrix = np.zeros((vocab_size, embedding_dim))

	with open(filepath, encoding="utf-8") as f:
		for line in f:
			word, *vector = line.split()
			if word in word_index:
				idx = word_index[word]
				embedding_matrix[idx] = np.array(
					vector, dtype=np.float32)[:embedding_dim]

	return embedding_matrix

def remove_underscores(text):
	words = text.split(' ')
	for word in words:
		if '_' in word:
			if '_emoj' not in word:
				words[words.index(word)] = ''
	return ' '.join(words)

def tweet_cleaner(text):
	soup = BeautifulSoup(text, 'lxml')
	souped = soup.get_text().lower()      # lowercase the whole thing here
	bomgone = souped.replace('ï¿½', ' ')
	re_cleaned = re_pat.sub(' ', bomgone)
	neg_handled = neg_pattern.sub(lambda x: contractions_dict[x.group()], re_cleaned)
	tokenized = tzer.tokenize(neg_handled)
	return " ".join(tokenized)

def train_model(model, family, target, X_train, X_test, y_train, y_test):
	# fit model
	model_lstm = model
	model_lstm.fit(train_X, y_train)

	model_lstm.model.save('lstm_model_'+target+'_new.h5')
	print('Saved model for target: ', target)


if __name__ == '__main__':
	file_cat1 = 'data/D1.tsv'
	file_cat2 = 'data/D2.tsv'
	file_cat3 = 'data/D3.tsv'
	dataframe_cat1 = pd.read_table(file_cat1)
	dataframe_cat2 = pd.read_table(file_cat2)
	dataframe_cat3 = pd.read_table(file_cat3)
	col1 = ['tweetID', 'text', 'relevant']
	col2 = ['tweetID', 'text', 'com_vape']
	col3 = ['tweetID', 'text', 'pro_vape']
	df_relevant = dataframe_cat1[col1]
	df_com = dataframe_cat2[col2]
	df_pro = dataframe_cat3[col3]

	x1 = df_relevant['text']
	x2 = df_com['text']
	x3 = df_pro['text']
	x = [x1, x2, x3]
	y = [df_relevant.relevant, df_com.com_vape, df_pro.pro_vape]

	# 9810 = relevant, 7006 = com_vape, 5677 for pro_vape
	#max_features = [10500, 7268, 5963]   
	max_features = [9810, 7006, 5677]
	maxlen = 75

	contractions_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}
	neg_pattern = re.compile(r'\b(' + '|'.join(contractions_dict.keys()) + r')\b')
	tzer = tokenize.RegexpTokenizer(r'[A-Za-z_]+')

	#remove url, unicodes, emojis
	pat1 = r'@[A-Za-z0-9]+'
	pat2 = r'https?://[^ ]+'
	pat3 = r'www\.[^ ]+'
	pat4 = r'\\u[^ ]+'
	combined_pat = r'|'.join((pat1, pat2, pat3, pat4))
	re_pat = re.compile(combined_pat)
	target = ['relevance', 'commercial', 'pro_vape']

	for i in range(0, len(x)):
		tweets = x[i].tolist()
		clean_tweets = []
		for t in tweets:
			tweet = tweet_cleaner(t)
			clean_tweets.append(remove_underscores(tweet))

		X_train, X_test, y_train, y_test = train_test_split(clean_tweets, y[i], test_size=0.1, random_state=42, stratify=y[i])

		tokens = []
		for text in clean_tweets:
			words = tokenize.word_tokenize(text)
			tokens.extend(words)

		lengths = []
		for text in clean_tweets:
			words = tokenize.word_tokenize(text)
			lengths.append(len(words))

		tokenizer = Tokenizer(num_words=max_features[i])
		tokenizer.fit_on_texts(list(X_train)+list(X_test))
		train_X = tokenizer.texts_to_sequences(X_train)
		test_X = tokenizer.texts_to_sequences(X_test)
		word_index = tokenizer.word_index
		train_X = pad_sequences(train_X, maxlen=maxlen)
		test_X = pad_sequences(test_X, maxlen=maxlen)

		with open('lstm_tokenizer_new.pickle', 'wb') as handle:
			pickle.dump(tokenizer, handle)

		embed_size = 200
		embedding_matrix = create_embedding_matrix('/pylon5/be5fpap/ancarey/AnnotationProjects/classification/embedding/glove.twitter.27B.200d.txt', tokenizer.word_index, 200)
		best_lstm = KerasClassifier(build_fn=create_lstm, max_features=max_features[i], embed_size=embed_size,
								max_len = 75, embedding_matrix = embedding_matrix, optimizer = 'adam',
								epochs = 5, batch_size=64)
		train_model(best_lstm, 'LSTM', target[i], X_train, X_test, y_train, y_test)
