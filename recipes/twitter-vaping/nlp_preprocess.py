# -*- coding: utf-8 -*-
"""
Version: 03-13-2019
Author: sanyabt
"""
import re, string, sys
from nltk import tokenize
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

'''
Replace emojis with text translations given in emoji list (emojis: dictionary of translated emojis).
'''
def emojify(text, emojis):
	text = str(text.encode('unicode-escape'))[2:-1]
	if '\\u' in text.lower():
		text = text.replace('\\\\U' , '\\\\u')
		text = text.replace('\\\\u' , ' \\\\u')
		words = text.split(' ')
		for word in words:
			if '\\u' in word:
				if word in emojis.keys():
					words[words.index(word)] = emojis[word]
				elif word[0:11] in emojis:
					word_1 = word[11:len(word)]
					words[words.index(word)] = emojis[word[0:11]] + ' ' + word_1
				elif word[0:7] in emojis:
					word_1 = word[7:len(word)]
					words[words.index(word)] = emojis[word[0:7]] + ' ' + word_1
		return ' '.join(words)
	return text

'''
Remove translated emojis from text from text.
'''
def emoji_remove(text):
	words = tokenize.word_tokenize(text)
	for word in words:
		if 'emoj_' in word:
			words[words.index(word)] = ''
	return ' '.join(words)

'''
Remove twitter metadata information (_url_, _mention_, _hashtag_ etc) from text (excluding emojis).
'''
def metadata_remove(text):
	words = tokenize.word_tokenize(text)
	for word in words:
		if '_' in word and 'neg_' not in word and 'emoj_' not in word:
			words[words.index(word)] = ''
	return ' '.join(words)

'''
Replace twitter url's, hashtags, unrecognized unicodes and mentions.
'''
def metadata_clean(text, url_pat, unicode_pat, mention_pat):
	#Handle URL's
	text = url_pat.sub('_url_', text)

	#Unidentified unicodes
	text = unicode_pat.sub('_unicode_', text)
	
	#Handle @mentions and hashtags
	text = mention_pat.sub('_mention_', text)
	text = text.replace('#', '_hashtag_')
	
	#remove extra backslack due to post-parse emojify, but dont want to remove the \ in other unicodes
	text = text.replace('\\', '')
	return text

'''
Expand the negation words in text using negation dictionary (defined below).
'''
def negation_expand(text, neg_pattern, negations_dic):
	
	#Expand negation contractions mentioned in negations dictionary
	text = neg_pattern.sub(lambda x: negations_dic[x.group()], text)
	return text

'''
Remove punctuation from text. Returns tokenized text either with or without punctuation.
'''
def punctuation_remove(text):

	tzer = tokenize.RegexpTokenizer(r'[A-Za-z0-9_]+')
	tokenized = tzer.tokenize(text)
	return ' '.join(tokenized)

'''
Remove digits from text.
'''
def digits_remove(text):
	result = ''.join(i for i in text if not i.isdigit())
	return result

def check_negation(token):
	flag = False
	if token in string.punctuation:
		flag = True
		return False, flag
	if '_' in token:
		return False, flag
	else:
		return True, flag

'''
Negation marking of text for all negation words defined below. 
1. Find all negation words in the text
2. Add NEG_ to tokens following the negation word till the next punctuation (end of sentence) - if punctuation present in next 4 tokens
3. Else add NEG_ to next 4 tokens (non-punctuation)
'''
def negation_marking(text, neg_mark_pattern):
	tokens = tokenize.word_tokenize(text)
	neg_matched = neg_mark_pattern.findall(' '.join(tokens))
	
	for item in neg_matched:
		if item in tokens:
			loc = tokens.index(item)
		
			if (len(tokens) - loc) <= 4:
				for tok in tokens[loc+1:]:
					ans, flag = check_negation(tok)
					if ans is True:
						tokens[tokens.index(tok)] = 'NEG_'+tok
					if flag is True:
						break
			else:
				for tok in tokens[loc+1:loc+5]:
					ans, flag = check_negation(tok)
					if ans is True:
						tokens[tokens.index(tok)] = 'NEG_'+tok
					if flag is True:
						break
	return ' '.join(tokens)

'''
Fix lengthening in text where consecutive similar characters occurring more than 2 times are reduced to 2.
'''
def normalize_text(text, pattern):
	return pattern.sub(r"\1\1", text)
	
'''
Remove stopwords from text using NLTK English stopwords list.
'''
def stopwords_remove(text):
	words = tokenize.word_tokenize(text)
	stop_words = set(stopwords.words('english'))
	words = [word for word in words if word.lower() not in stop_words]
	return ' '.join(words)

'''
Porter stemming algorithm applied to the text. Note: converts text to lowercase and also stem stopwords (such as 'was' to 'wa'). Do with caution.
'''
def stemming_apply(text, stemmer):
	tokens = tokenize.word_tokenize(text)
	stems = []
	for t in tokens:
		stems.append(stemmer.stem(t))
	return ' '.join(stems)

'''
Based on text options specified, run the pipeline and process tweets text.
'''
def preprocess(tweets, text_options):

	#Translate emojis for all tweets (this is the default in parsing)
	emojis = {}
	with open('data/emojilist5.csv', 'r') as f:
		for line in f:
			unic=line.split(',')[0].lower()
			trans=line.split(',')[1]
			emojis[unic]=trans
	tweets = tweets.apply(emojify, args=(emojis,))
	
	#Metadata information clean for all tweets
	pat1 = r'https?://[A-Za-z0-9./]+'
	pat2 = r'www\\.[^ ]+'
	combined_pat = r'|'.join((pat1, pat2))
	url_pat = re.compile(combined_pat)

	pat3 = r'\\u[^ ]+'
	unicode_pat = re.compile(pat3)

	pat4 = r'@[A-Za-z0-9_]+'
	mention_pat = re.compile(pat4)
	tweets = tweets.apply(metadata_clean, args=(url_pat, unicode_pat, mention_pat))

	#Expand negations
	if text_options['negation_expand'] is True:
		negations_dic = {"isn\'t" : "is not",
				"aren\'t" : "are not",
				"wasn\'t" : "was not",
				"weren\'t" : "were not",
				"haven\'t" : "have not",
				"hasn\'t" : "has not",
				"hadn\'t" : "had not",
				"won\'t" : "will not",
				"wouldn\'t" : "would not",
				"don\'t" : "do not",
				"doesn\'t" : "does not",
				"didn\'t" : "did not",
				"can\'t" : "can not",
				"couldn\'t" : "could not",
				"shouldn\'t" : "should not",
				"mightn\'t" : "might not",
				"mustn\'t" : "must not",
				"shan\'t" : "shall not",
				"ain\'t" : "am not"}
	
		neg_expand_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')
		tweets = tweets.apply(negation_expand, args=(neg_expand_pattern, negations_dic))
	
	#Remove punctuation
	if text_options['punctuation_remove'] is True:
		tweets = tweets.apply(punctuation_remove)

	#Remove metadata- hashtags, urls, mentions, unicode
	if text_options['metadata_remove'] is True:
		tweets = tweets.apply(metadata_remove)

	#Remove emojis from tweets
	if text_options['emoji_remove'] is True:
		tweets = tweets.apply(emoji_remove)

	#Remove digits
	if text_options['digits_remove'] is True:
		tweets = tweets.apply(digits_remove)
	
	#Mark negations
	if text_options['negation_mark'] is True:
		neg_words = ['not', 'never', 'no', 'nothing', 'noone', 'nowhere', 'none',
				'isnt', 'arent', 'wasnt', 'werent', 'havent', 'hasnt', 'hadnt',
				'wont', 'wouldnt', 'dont', 'doesnt', 'didnt', 'cant', 'couldnt',
				'shouldnt', 'mightnt', 'mustnt', 'shant', 'aint']
		neg_mark_pattern = re.compile(r'\b(' + '|'.join(neg_words) + r')\b')
		tweets = tweets.apply(negation_marking, args=(neg_mark_pattern,))
	
	#Normalization
	if text_options['normalize'] is True:
		repeat_pattern = re.compile(r"(.)\1{2,}")
		tweets = tweets.apply(normalize_text, args=(repeat_pattern,))

	#Remove stopwords: done before stemming and after negation marking (stemmer stems stopwords, if done before negation marking, some negation words removed)
	if text_options['stopwords_remove'] is True:
		tweets = tweets.apply(stopwords_remove)

	#Stemming. Note: will convert to lowercase and also stem stopwords (such as 'was' to 'wa')
	if text_options['stemming'] is True:
		ps = PorterStemmer()
		tweets = tweets.apply(stemming_apply, args=(ps,))

	#Lowercasing of text
	if text_options['lower'] is True:
		tweets = tweets.str.lower()

	return tweets
