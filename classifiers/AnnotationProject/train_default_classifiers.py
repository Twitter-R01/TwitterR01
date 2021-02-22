import pickle
import sys
import pandas as pd
import numpy as np

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, brier_score_loss
from sklearn.pipeline import Pipeline
from time import time
import nlp_preprocess
import inspect
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.metrics import make_scorer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

def set_classifier(model):
    if model == "SVM":
        return SVC(kernel='linear',probability=True)
    elif model == "RF":
        return RandomForestClassifier()
    elif model == "NB":
        return MultinomialNB()
    else:
        return LogisticRegression()


def train_default(file, target = 'relevant', classifier = LogisticRegression()):
    f = file
    df = pd.read_table(f)
    col = ['tweetID', 'text']
    col.append(target)
    df = df[col]
    x = df.text
    y = df[str(target)]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42, stratify=y)
    model = set_classifier(classifier)
    pipe = Pipeline(
        steps=[
        ('vectorizer', CountVectorizer()),
        ('clf', model)
        ]
    )
    default = pipe.fit(x_train, y_train)
    y_prob = default.predict_proba(x_test)
    return roc_auc_score(y_test, y_prob[:,1])

if __name__ == '__main__':
    default_score = train_default(sys.argv[1], sys.argv[2], sys.argv[3])
    print('=====================')
    print(sys.argv[3], ": ", sys.argv[2])
    print('---------------------')
    print('Test score: ', round(default_score, 3))
    print('=====================')
