import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, brier_score_loss
from sklearn.pipeline import Pipeline
from time import time
import nlp_preprocess
import inspect
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.metrics import make_scorer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

class ClassifierPipeline(BaseEstimator):

	def __init__(self, estimator = LogisticRegression(),):
		"""
		A custom BaseEstimator that can switch between classifiers in the pipe.
		Defaults to Logistic Regression.
		
		:param estimator: sklearn object; switches between any sklearn estimator
		"""
		self.estimator = estimator
	
	def fit(self, X, y=None, **kwargs):
		self.estimator.fit(X, y)
		return self
	
	def predict(self, X, y=None):
		return self.estimator.predict(X)
	
	def predict_proba(self, X):
		return self.estimator.predict_proba(X)
	
	def score(self, X, y):
		return self.estimator.score(X, y)

class NLP_transformer(BaseEstimator, TransformerMixin):
	def __init__(self, 
				 metadata_remove=True, emoji_remove=True, negation_expand=True, punctuation_remove=True, 
				 digits_remove=True, negation_mark=True, normalize=True, stemming=True, 
				 stopwords_remove=True, lower=True):
		"""
		A custom Transformer that takes in text column of dataframe and transforms to clean text.
		Arguments are text options specified above.
		"""
		args, _, _, values = inspect.getargvalues(inspect.currentframe())
		values.pop('self')
		self.text_options = {}
		for arg, val in values.items():
			self.text_options[arg] = val
	def transform(self, X, y=None):
		return nlp_preprocess.preprocess(X, self.text_options)
		
	def fit(self, X, y=None):
		return self

if __name__ == '__main__':

	#Custom define range for number of features
	# min num_features
	a = 50
	r = 2
	# length of sequence
	length = 10
	sequence = [a * r ** (n-1) for n in range(1, length + 1)]
	max_features_params = [x for x in sequence if x <= 3500]

	pipe = Pipeline(
	steps=[
	('preprocess', NLP_transformer()), 
	('vectorizer', CountVectorizer(lowercase=False)), 
	('clf', ClassifierPipeline())
	]
	)

	scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score),
		   'Brier': make_scorer(brier_score_loss), 'f1-score': make_scorer(f1_score),
		   'precision': make_scorer(precision_score), 'recall': make_scorer(recall_score)}

	param_grid = [
	{
		'preprocess__metadata_remove': [False, True],
		'preprocess__emoji_remove': [False, True],
		'preprocess__punctuation_remove': [False, True],
		'preprocess__negation_expand': [False, True],
		'preprocess__digits_remove': [False, True],
		'preprocess__negation_mark': [False, True],
		'preprocess__normalize': [False, True],
		'preprocess__stopwords_remove': [False, True],
		'preprocess__stemming': [False, True],
		'preprocess__lower': [False, True],
		'clf__estimator': [LogisticRegression()],
		'vectorizer__max_features': max_features_params,
		'vectorizer__ngram_range': [(1, 1), (1, 2), (1, 3)],
		'vectorizer__stop_words': [None],
		'clf__estimator__penalty': ['l2'],
		'clf__estimator__C': [0.001, 0.01, 0.1, 1, 10],
	},
	]

	file_i = '/home/sanyabt/AnnotationProjects/classification/data/D1.tsv'
	dataframe = pd.read_table(file_i)
	col = ['tweetID', 'text', 'relevant']
	df = dataframe[col]
	print(df.info())
	x = df.text
	y = df.relevant

	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42, stratify=y)

	search = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1, return_train_score=False, verbose=3, scoring=scoring, refit='AUC')
	search.fit(x_train, y_train)

	print('SEARCH')
	print('Best estimator: ', search.best_estimator_)
	print('Best score: ', search.best_score_)
	print('Best parameters: ', search.best_params_)

	results = search.cv_results_
	with open('classifier_results_4k.pickle', 'wb') as file_o:
		pickle.dump(results, file_o)

	y_pred = search.predict(x_test)
	test_score = search.score(x_test, y_test)
	print('Test Score: ', test_score)
	print('Test set accuracy: ', accuracy_score(y_test, y_pred))

	data = {'x_test': x_test, 'y_test': y_test, 'y_pred': y_pred}
	output_df = pd.DataFrame(data)
	output_df.to_csv('data/classifier_results_test_set_4k.tsv', sep='\t', index=False)