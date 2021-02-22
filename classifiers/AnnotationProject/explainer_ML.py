from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score, f1_score, accuracy_score, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import nlp_preprocess
import inspect
import pickle
import shap
import matplotlib.pyplot as plt
plt.switch_backend('agg')

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
		print(self.text_options)
	def transform(self, X, y=None):
		return nlp_preprocess.preprocess(X, self.text_options)

	def fit(self, X, y=None):
		return self

def explain_model(model, nlp_args, X_train, X_test, y_train, y_test):
	model.fit(X_train, y_train)
	# we use the first 100 training examples as our background dataset to integrate over
	if pipe.named_steps['clf'].__class__.__name__ == "LogisticRegression":
		explainer = shap.LinearExplainer(model.named_steps['clf'], model.named_steps['vectorizer'].transform(model.named_steps['preprocess'].transform(X_train)))
		shap_values = explainer.shap_values(model.named_steps['vectorizer'].transform(model.named_steps['preprocess'].transform(X_test)).toarray())
	elif pipe.named_steps['clf'].__class__.__name__ == "RandomForestClassifier":
		explainer = shap.TreeExplainer(model.named_steps['clf'])
		shap_values = explainer.shap_values(model.named_steps['vectorizer'].transform(model.named_steps['preprocess'].transform(X_test)).toarray())
	else:
		X_train_summary = shap.kmeans(model.named_steps['vectorizer'].transform(model.named_steps['preprocess'].transform(X_train)).toarray(), 10)
		explainer = shap.KernelExplainer(model.named_steps['clf'].predict_proba, X_train_summary)
		shap_values = explainer.shap_values(model.named_steps['vectorizer'].transform(model.named_steps['preprocess'].transform(X_test)).toarray())

	return explainer, shap_values

family_name = ['LogisticRegression', 'RandomForest', 'SVM', 'MultinomialNB']
with open("/pylon5/be5fpap/kurtosis/classifiers/ML_params.pkl", "rb") as f:
	nlp_params, model_params, vectorizer_params, models = pickle.load(f)

file = '/pylon5/be5fpap/kurtosis/classifiers/data/D1.tsv'
dataframe = pd.read_table(file)
col = ['tweetID', 'text', 'relevant']
df = dataframe[col]
print(df.info())
x = df.text
y = df.relevant

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42, stratify=y)

for i in range(0, 4):
	pipe = Pipeline(
		steps=[
			('preprocess', NLP_transformer(**nlp_params[i])),
			('vectorizer', CountVectorizer(**vectorizer_params[i], lowercase=False, stop_words=None)),
			('clf', models[i])
		]
	)
	explainer, shap_values = explain_model(pipe, nlp_params[i], X_train, X_test, y_train, y_test)
	with open('explain/explainer_%s_%s.pkl' % (family_name[i], col[2]), 'wb') as f:
		pickle.dump(explainer, f)
		pickle.dump(shap_values, f)
	shap.summary_plot(shap_values, pipe.named_steps['vectorizer'].transform(pipe.named_steps['preprocess'].transform(X_test)).toarray(), plot_type='bar', feature_names = pipe.named_steps['vectorizer'].get_feature_names(), show=False)
	f = plt.gcf()
	f.savefig('explain/shap_summary_%s_%s.png' % (family_name[i], col[2]), dpi=300, bbox_inches='tight')
	plt.close()
