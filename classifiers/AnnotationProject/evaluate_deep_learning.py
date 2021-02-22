import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
sns.set()
import gensim
import multiprocessing
import tensorflow as tf
import re
from nltk import tokenize
from bs4 import BeautifulSoup
from gensim.models import KeyedVectors
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn import utils
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, precision_score, recall_score, precision_recall_curve, f1_score, accuracy_score, brier_score_loss
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import LSTM, Conv1D, Activation, Bidirectional, Dense, Dropout, Input, Embedding, Flatten, GlobalMaxPooling1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.callbacks import Callback, EarlyStopping
import random as rn

os.environ['PYTHONHASHSEED'] = '0'
rn.seed(4959)
np.random.seed(4959)
tf.set_random_seed(4959)

# Force TensorFlow to use single thread.
# Multiple threads are a potential source of
# non-reproducible results.
# For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from keras import backend as K

# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


def calibration_plot(model, X_train, X_test, y_train, y_test):
	model = model
	model.fit(X_train, y_train)
	y_prob = model.predict_proba(X_test)[:,1]
	fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_prob, strategy='quantile', n_bins=10)
	return fraction_of_positives, mean_predicted_value, y_prob

def create_bilstm(max_features, embed_size, max_len, embedding_matrix, optimizer='rmsprop'):
	inputs = Input(shape=(maxlen,))
	x = Embedding(max_features, embed_size, weights=[embedding_matrix],trainable=False)(inputs)
	'''
	Here 64 is the size(dim) of the hidden state vector as well as the output vector. Keeping return_sequence we want the output for the entire sequence. So what is the dimension of output for this layer?
	    64*70(maxlen)*2(bidirection concat)
	CuDNNLSTM is fast implementation of LSTM layer in Keras which only runs on GPU
	'''
	x = Bidirectional(LSTM(64, return_sequences=True))(x)
	conc = Dense(64, activation="relu")(x)
	conc = Dropout(0.5)(conc)
	conc = Flatten()(conc)
	outp = Dense(1, activation="sigmoid")(conc)
	model = Model(inputs=inputs, outputs=outp)
	model.compile(loss='binary_crossentropy', optimizer=optimizer)
	return model

def create_cnn(max_features, embed_size, max_len, embedding_matrix, filters=25, kernel_size=1, optimizer='rmsprop'):
	inputs = Input(name='inputs', shape=(max_len,))
	print(inputs.shape)
	layer = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inputs)
	layer = Conv1D(filters=filters, kernel_size=kernel_size, padding='valid', strides=1)(layer)
	layer = Activation('relu')(layer)
	layer = GlobalMaxPooling1D()(layer)
	layer = Dense(256)(layer)
	layer = Activation('relu')(layer)
	layer = Dense(1,name='out_layer')(layer)
	layer = Activation('sigmoid')(layer)
	model = Model(inputs=inputs,outputs=layer)
	model.compile(loss='binary_crossentropy', optimizer=optimizer)
	return model

def create_embedding_matrix(filepath, word_index, embedding_dim):
	vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
	embedding_matrix = np.zeros((vocab_size, embedding_dim))

	with open(filepath) as f:
		for line in f:
			word, *vector = line.split()
			if word in word_index:
				idx = word_index[word]
				embedding_matrix[idx] = np.array(
					vector, dtype=np.float32)[:embedding_dim]

	return embedding_matrix

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

def create_lstm_cnn(max_features, embed_size, max_len, embedding_matrix,
					optimizer='rmsprop', filters=25, kernel_size=1):
	inputs = Input(name='inputs', shape=(max_len,))
	layer = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inputs)
	layer = LSTM(64, return_sequences=True)(layer)
	layer = Conv1D(filters=filters, kernel_size=kernel_size, padding='valid', strides=1)(layer)
	layer = Activation('relu')(layer)
	layer = MaxPooling1D(pool_size=2, padding='valid')(layer)
	layer = Flatten()(layer)
	layer = Dropout(0.5)(layer)
	layer = Dense(1,name='out_layer')(layer)
	layer = Activation('sigmoid')(layer)
	model = Model(inputs=inputs,outputs=layer)
	model.compile(loss='binary_crossentropy', optimizer='rmsprop')
	return model

def evaluate_model(model, family, target, X_train, X_test, y_train, y_test):
	# fit model
	model = model
	model.fit(train_X, y_train)
	y_prob = model.predict_proba(test_X)[:,1]
	param_dict = model.get_params()
	del param_dict['build_fn']
	del param_dict['embedding_matrix']

	# calculate youden's J statistic
	fpr, tpr, thresholds = roc_curve(y_test, y_prob)
	precision, recall, _ = precision_recall_curve(y_test, y_prob)
	auc = roc_auc_score(y_test, y_prob)
	J = thresholds[np.argmax(tpr - fpr)]

	d = {'target': [target],
	'model': [family],
	'params': [param_dict],
	'precision': [precision_score(y_test, model.predict(test_X))],
	'recall': [recall_score(y_test, model.predict(test_X))],
	'f1': [f1_score(y_test, model.predict(test_X))],
	'accuracy': [accuracy_score(y_test, model.predict(test_X))],
	'auroc': [roc_auc_score(y_test, model.predict_proba(test_X)[:,1])],
	'auprc': [average_precision_score(y_test, model.predict_proba(test_X)[:,1])],
	'brier': [brier_score_loss(y_test, model.predict_proba(test_X)[:,1])],
	'youden': [J],
	'precision_J': [precision_score(y_test, np.where(model.predict_proba(test_X)[:,1] > J, 1, 0))],
	'recall_J': [recall_score(y_test, np.where(model.predict_proba(test_X)[:,1] > J, 1, 0))],
	'f1_J': [f1_score(y_test, np.where(model.predict_proba(test_X)[:,1] > J, 1, 0))],
	'accuracy_J': [accuracy_score(y_test, np.where(model.predict_proba(test_X)[:,1] > J, 1, 0))]
	}
	auc_plot = {'classifiers': get_label(model), 'fpr':fpr, 'tpr':tpr, 'auc':auc}
	prc_plot = {'classifiers': get_label(model), 'precision': precision, 'recall': recall}
	pred_df = {'y_true': y_test, 'y_pred': y_prob}
	return(pd.DataFrame(data=d), auc_plot, prc_plot, pd.DataFrame(data=pred_df))

def get_label(model):
	if model.build_fn.__name__ == "create_cnn":
		return 'CNN'
	elif model.build_fn.__name__ == "create_lstm":
		return 'LSTM'
	elif model.build_fn.__name__ == "create_lstm_cnn":
		return 'LSTM-CNN'
	elif model.build_fn.__name__ == "create_bilstm":
		return 'BiLSTM'
	else:
		return 'unknown'

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

if __name__ == '__main__':
	# edit settings
	file_i = 'data/D3.tsv'
	dataframe = pd.read_table(file_i)
	col = ['tweetID', 'text', 'pro_vape']
	df = dataframe[col]
	x = df.text
	y = df.pro_vape
	embedding = "vape"
	if embedding == "vape":
		# static across all labels
		max_features = 166395
	else:
		# 9810 = relevant, 7006 = com_vape, 5677 for pro_vape
		max_features = 5677
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

	tweets = x.tolist()
	clean_tweets = []
	for t in tweets:
		tweet = tweet_cleaner(t)
		clean_tweets.append(remove_underscores(tweet))
	x = df.text.apply(tweet_cleaner).apply(remove_underscores)
	X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42, stratify=y)

	tokens = []
	for text in clean_tweets:
		words = tokenize.word_tokenize(text)
		tokens.extend(words)

	lengths = []
	for text in clean_tweets:
		words = tokenize.word_tokenize(text)
		lengths.append(len(words))

	tokenizer = Tokenizer(num_words=max_features)
	tokenizer.fit_on_texts(list(X_train)+list(X_test))
	train_X = tokenizer.texts_to_sequences(X_train)
	test_X = tokenizer.texts_to_sequences(X_test)
	word_index = tokenizer.word_index
	train_X = pad_sequences(train_X, maxlen=maxlen)
	test_X = pad_sequences(test_X, maxlen=maxlen)

	if embedding == "vape":
		bow_model = KeyedVectors.load('embedding/word2vec_cbow.model')
		sg_model = KeyedVectors.load('embedding/word2vec_sg.model')
		embed_size = 300
		embedding_matrix = np.concatenate((bow_model.wv.vectors, sg_model.wv.vectors), axis=1)
	else:
		embed_size = 200
		embedding_matrix = create_embedding_matrix(
		'embedding/glove.twitter.27B.200d.txt', tokenizer.word_index, 200
		)

	best_cnn = KerasClassifier(build_fn=create_cnn, max_features=max_features, embed_size=embed_size, max_len = 75,
							   embedding_matrix = embedding_matrix, optimizer='adam', filters = 100,
							   kernel_size=3, epochs=10, batch_size=32)
	auc_df = pd.DataFrame(columns=['classifiers', 'fpr', 'tpr', 'auc'])
	prc_df = pd.DataFrame(columns=['classifiers', 'precision', 'recall'])
	results = pd.DataFrame(columns=['target', 'model', 'params', 'precision', 'recall', 'f1', 'accuracy', 'auroc',
									'brier', 'youden', 'precision_J', 'recall_J',
									'f1_J', 'accuracy_J'])
	df, auc_tmp, prc_tmp, preds_df = evaluate_model(best_cnn, 'CNN', 'commercial', X_train, X_test, y_train, y_test)
	auc_df = auc_df.append(auc_tmp, ignore_index=True)
	prc_df = prc_df.append(prc_tmp, ignore_index=True)
	preds_df.to_csv('results/cnn_%s_%s_preds.csv' % (embedding, col[2]))
	results = results.append(df)

	best_lstm = KerasClassifier(build_fn=create_lstm, max_features=max_features, embed_size=embed_size,
								max_len = 75, embedding_matrix = embedding_matrix, optimizer = 'adam',
								epochs = 5, batch_size=64)
	df, auc_tmp, prc_tmp, preds_df = evaluate_model(best_lstm, 'LSTM', 'commercial', X_train, X_test, y_train, y_test)
	auc_df = auc_df.append(auc_tmp, ignore_index=True)
	prc_df = prc_df.append(prc_tmp, ignore_index=True)
	preds_df.to_csv('results/lstm_%s_%s_preds.csv' % (embedding, col[2]))
	results = results.append(df)

	best_lstm_cnn = KerasClassifier(build_fn=create_lstm_cnn, max_features=max_features, embed_size=embed_size,
								max_len = 75, embedding_matrix = embedding_matrix, optimizer = 'rmsprop',
								filters = 25, kernel_size = 2, epochs = 5, batch_size = 16)
	df, auc_tmp, prc_tmp, preds_df = evaluate_model(best_lstm_cnn, 'LSTM-CNN', 'commercial', X_train, X_test, y_train, y_test)
	auc_df = auc_df.append(auc_tmp, ignore_index=True)
	prc_df = prc_df.append(prc_tmp, ignore_index=True)
	preds_df.to_csv('results/lstm_cnn_%s_%s_preds.csv' % (embedding, col[2]))
	results = results.append(df)

	best_bilstm = KerasClassifier(build_fn=create_bilstm, max_features=max_features, embed_size=embed_size,
								max_len = 75, embedding_matrix = embedding_matrix, optimizer = 'adam',
								epochs = 10, batch_size = 16)
	df, auc_tmp, prc_tmp, preds_df = evaluate_model(best_bilstm, 'BiLSTM', 'commercial', X_train, X_test, y_train, y_test)
	auc_df = auc_df.append(auc_tmp, ignore_index=True)
	prc_df = prc_df.append(prc_tmp, ignore_index=True)
	preds_df.to_csv('results/bilstm_%s_%s_preds.csv' % (embedding, col[2]))
	results = results.append(df)

	fig = plt.figure(figsize=(10, 10))
	ax = plt.subplot()
	ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

	for model in [best_cnn, best_lstm, best_lstm_cnn, best_bilstm]:
		fraction_of_positives, mean_predicted_value, y_prob = calibration_plot(model, train_X, test_X, y_train, y_test)
		ax.plot(mean_predicted_value, fraction_of_positives, "s-",
				label="%s" % (get_label(model)))

	ax.legend(loc="lower right")
	ax.set_ylabel("Fraction of positives")
	ax.set_xlabel("Mean predicted value")
	ax.set_ylim([-0.05, 1.05])
	ax.set_title('Calibration plots for %s classifiers with %s word embeddings' % (col[2], embedding))
	plt.tight_layout()
	plt.savefig('results/calibration_plot_DL_%s_%s.png' % (embedding, col[2]))

	auc_df.set_index('classifiers', inplace=True)

	fig = plt.figure(figsize=(8,6))
	for i in auc_df.index:
		plt.plot(auc_df.loc[i]['fpr'],
				auc_df.loc[i]['tpr'],
				label="{}, AUC={:.3f}".format(str(i), auc_df.loc[i]['auc']))
	plt.plot([0,1], [0,1], color='orange', linestyle='--')

	plt.xticks(np.arange(0.0, 1.1, step=0.1))
	plt.xlabel("False Positive Rate", fontsize=15)

	plt.yticks(np.arange(0.0, 1.1, step=0.1))
	plt.ylabel("True Positive Rate", fontsize=15)

	plt.text(x = -0.05, y = 1.13, s = 'ROC Curve Analysis', fontsize=15, weight='bold', alpha=1.0)
	plt.text(x = -0.05, y = 1.1, s = '%s classifiers with %s word embeddings' % (col[2], embedding), fontsize=10)
	plt.legend(prop={'size':13}, loc='lower right')

	plt.savefig('results/AUC_%s_%s.png' % (embedding, col[2]))

	prc_df.set_index('classifiers', inplace=True)

	fig = plt.figure(figsize=(8,6))
	for i in prc_df.index:
		plt.plot(prc_df.loc[i]['recall'],
		prc_df.loc[i]['precision'],
		label="{}".format(str(i)))

	plt.xticks(np.arange(0.0, 1.1, step=0.1))
	plt.xlabel("Recall", fontsize=15)

	plt.yticks(np.arange(0.0, 1.1, step=0.1))
	plt.ylabel("Precision", fontsize=15)

	plt.text(x = -0.05, y = 1.13, s = 'PRC Curve Analysis', fontweight='bold', fontsize=15)
	plt.text(x = -0.05, y = 1.1, s = '%s classifiers with %s word embeddings' % (col[2], embedding), fontsize=10)
	plt.legend(prop={'size':13}, loc='lower right')
	plt.savefig('results/PRC_%s_%s.png' % (embedding, col[2]))

	results.to_csv('./NN_results_%s_%s.csv' % (embedding, col[2]), index=False)
