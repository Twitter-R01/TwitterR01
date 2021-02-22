#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import sys, os
import pickle
import numpy as np
import multiprocessing
import tensorflow as tf
import re
from nltk import tokenize
from bs4 import BeautifulSoup
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

def classify(model, tweets):
		y_pred = model.predict(tweets)
		return y_pred

def remove_underscores(text):
	words = text.split(' ')
	for word in words:
		if '_' in word:
			if '_emoj' not in word:
				words[words.index(word)] = ''
	return ' '.join(words)

def tweet_cleaner(text):
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
	soup = BeautifulSoup(text, 'lxml')
	souped = soup.get_text().lower()      # lowercase the whole thing here
	bomgone = souped.replace('ï¿½', ' ')
	re_cleaned = re_pat.sub(' ', bomgone)
	neg_handled = neg_pattern.sub(lambda x: contractions_dict[x.group()], re_cleaned)
	tokenized = tzer.tokenize(neg_handled)
	return " ".join(tokenized)

if __name__ == '__main__':

	#load models
	model_path = '/home/ancarey/AnnotationProjects/classification/'
	lstm_relevance = load_model(model_path+'lstm_model_relevance_new.h5')
	lstm_commercial = load_model(model_path+'lstm_model_commercial_new.h5')
	lstm_pro_vape = load_model(model_path+'lstm_model_pro_vape_new.h5')

	with open('/home/ancarey/AnnotationProjects/classification/lstm_tokenizer_new.pickle', 'rb') as handle:
		tokenizer = pickle.load(handle)

	#load files
	dir_path = '/pylon5/be5fpap/jcolditz/parser_out/juul_fda/juul_classification_20180301-20181231/'
	maxlen = 75
	start = None
	end = None

	#for saving counts of tweets after classification - can edit
	save = True
	threshold = 'custom'
	
	if threshold == 'default':
		threshold_values = [0.5, 0.5, 0.5]
	else:
		threshold_values = [0.847, 0.344, 0.837]

	count_dict = {'date': [],
	'original_tweets': [],
	'relevant': [],
	'commercial': [],
	'non_commercial': [],
	'pro_vape': [],
	'not_pro': []}

	if len(sys.argv) > 1: # If command line arguments were passed
		i = 0
		for arg in sys.argv:
			# '-d' indicates start/end dates (req. 2 MMDDYYYY objects)
			if arg.lower() in ['-d','-date','-dates']:
				start = int(sys.argv[i+2])
				end = int(sys.argv[i+3])

	files = os.listdir(dir_path)
	files.sort()
	print('\n\nSTART_DATE: '+str(start))
	print('END_DATE:   '+str(end))

	for file in files:
		if file[-4:] == '.tsv':
			if int(file[:8]) >= int(start):
				if int(file[:8]) <=int(end):
					df = pd.read_table(dir_path+file)

					#df = dataframe.loc[dataframe.re_t_id.isna()]
					#find and remove duplicate tweet IDs before running

					df = df.drop_duplicates()
					texts = df['t_text'] 
					predict_tweets = tokenizer.texts_to_sequences(texts)
					predict_tweets = pad_sequences(predict_tweets, maxlen=maxlen)

					#create df for results
					results = df
					results['relevant'] = classify(lstm_relevance, predict_tweets)
					results['com_vape'] = -1
					results['pro_vape'] = -1

					#match tweetID's and insert commercial results into results df
					relevant_tweets = results.loc[results.relevant > threshold_values[0], 't_text']
					predict_tweets = tokenizer.texts_to_sequences(relevant_tweets)
					predict_tweets = pad_sequences(predict_tweets, maxlen=maxlen)
					results.loc[results.relevant > threshold_values[0], 'com_vape'] = classify(lstm_commercial, predict_tweets)

					#match tweetID's and insert pro_vape results into results df
					non_commercial_tweets = results.loc[(results.com_vape <= threshold_values[1]) & (results.com_vape >= 0), 't_text']
					predict_tweets = tokenizer.texts_to_sequences(non_commercial_tweets)
					predict_tweets = pad_sequences(predict_tweets, maxlen=maxlen)				
					results.loc[(results.com_vape <= threshold_values[1]) & (results.com_vape >= 0), 'pro_vape'] = classify(lstm_pro_vape, predict_tweets)

					'''print('Saving results of file: ', str(file))
					results.to_csv('/pylon5/be5fpap/jcolditz/parser_out/juul_fda/juul_classification_20180301-20181231/classification_output/'+str(file[:14])+'.tsv', sep='\t', index=False)
					'''
					#save counts to dictionary
					if save:
						count_dict['date'].append(int(file[:8]))
						count_dict['original_tweets'].append(len(df.index))

						relevant_count = results.loc[results.relevant > threshold_values[0]]
						commercial_count = results.loc[results.com_vape > threshold_values[1]]
						non_commercial_count = results.loc[(results.com_vape <= threshold_values[1]) & (results.com_vape >= 0)]
						pro_vape_count = results.loc[results.pro_vape > threshold_values[2]]
						not_pro_count = results.loc[(results.pro_vape <= threshold_values[2]) & (results.pro_vape >= 0)]
						count_dict['relevant'].append(len(relevant_count.index))
						count_dict['commercial'].append(len(commercial_count.index))
						count_dict['non_commercial'].append(len(non_commercial_count.index))
						count_dict['pro_vape'].append(len(pro_vape_count.index))
						count_dict['not_pro'].append(len(not_pro_count.index))

	count_df = pd.DataFrame(data=count_dict)
	count_df = count_df.groupby(['date']).sum()	
	print('Saving counts to file')
	
	count_df.to_csv('/pylon5/be5fpap/ancarey/AnnotationProjects/classification/verify_results/verify_classifier.tsv', sep='\t', columns = ['original_tweets', 'relevant', 'commercial', 'non_commercial', 'pro_vape', 'not_pro'])
	#count_df.to_csv('/pylon5/be5fpap/jcolditz/parser_out/juul_fda/juul_classification_20180301-20181231/classification_output/tweet_frequency_custom_2018.tsv', sep='\t', columns = ['original_tweets', 'relevant', 'commercial', 'non_commercial', 'pro_vape', 'not_pro'])

