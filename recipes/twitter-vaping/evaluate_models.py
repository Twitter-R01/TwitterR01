from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, precision_score, recall_score, f1_score, accuracy_score, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Added this to fix "Invalid DISPLAY variable" error 
import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import pandas as pd
import numpy as np
import nlp_preprocess
import inspect
import pickle


dir_in = '/pylon5/be5fpap/jcolditz/UArk/recipes/twitter-vaping/'

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

def calibration_plot(model, X_train, X_test, y_train, y_test):
    model = model
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:,1]
    try:
        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_prob, strategy='quantile', n_bins=10)
    except:
        print('TypeError: calibration_curve() got an unexpected keyword argument \'strategy\'\n',
              'running calibration_curve without this argument')
        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_prob, n_bins=10)
    return fraction_of_positives, mean_predicted_value, y_prob

def get_label(model):
    if model.named_steps['clf'].__class__.__name__ == "LogisticRegression":
        return 'LR'
    elif model.named_steps['clf'].__class__.__name__ == "RandomForestClassifier":
        return 'RF'
    elif model.named_steps['clf'].__class__.__name__ == "CalibratedClassifierCV":
        return 'SVM'
    elif model.named_steps['clf'].__class__.__name__ == "MultinomialNB":
        return 'NB'
    else:
        return 'unknown'

def evaluate_model(model, target, nlp_args, X_train, X_test, y_train, y_test):
    # fit model
    model = model
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:,1]
    # calculate youden's J statistic
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    J = thresholds[np.argmax(tpr - fpr)]

    d = {'preprocess': [str(nlp_args)],
    'vectorizer': [str(model.steps[1])],
    'algorithm': [str(model.steps[2])],
    'target': [target],
    'precision': [precision_score(y_test, model.predict(X_test))],
    'recall': [recall_score(y_test, model.predict(X_test))],
    'f1': [f1_score(y_test, model.predict(X_test))],
    'accuracy': [accuracy_score(y_test, model.predict(X_test))],
    'auroc': [roc_auc_score(y_test, model.predict_proba(X_test)[:,1])],
    'auprc': [average_precision_score(y_test, model.predict_proba(X_test)[:,1])],
    'brier': [brier_score_loss(y_test, model.predict_proba(X_test)[:,1])],
    'youden': [J],
    'precision_J': [precision_score(y_test, np.where(model.predict_proba(X_test)[:,1] > J, 1, 0))],
    'recall_J': [recall_score(y_test, np.where(model.predict_proba(X_test)[:,1] > J, 1, 0))],
    'f1_J': [f1_score(y_test, np.where(model.predict_proba(X_test)[:,1] > J, 1, 0))],
    'accuracy_J': [accuracy_score(y_test, np.where(model.predict_proba(X_test)[:,1] > J, 1, 0))]
    }

    auc_plot = {'classifiers': get_label(model), 'fpr':fpr, 'tpr':tpr, 'auc':auc}
    prc_plot = {'classifiers': get_label(model), 'precision': precision, 'recall': recall}
    pred_df = {'y_true': y_test, 'y_pred': y_prob}
    with open(dir_in+'classifiers/'+target+'_'+get_label(model)+'.pickle', 'wb+') as file_t:
        pickle.dump(model, file_t)    
    return(pd.DataFrame(data=d), auc_plot, prc_plot, pd.DataFrame(data=pred_df))

if __name__ == '__main__':
    file = dir_in+'data/D1.tsv'

    family_name = ['LogisticRegression', 'RandomForest', 'SVM', 'MultinomialNB']

    nlp_params = [{
    'metadata_remove': False, 'emoji_remove': False, 'negation_expand': False,
    'punctuation_remove': False, 'digits_remove': False,
    'negation_mark': False, 'normalize': False, 'stemming': False,
    'stopwords_remove': False, 'lower': False
}, {
    'metadata_remove': False, 'emoji_remove': True, 'negation_expand': True,
    'punctuation_remove': True, 'digits_remove': True,
    'negation_mark': True, 'normalize': True, 'stemming': False, 'stopwords_remove': True,
    'lower': True
},
{
    'metadata_remove': True, 'emoji_remove': False, 'negation_expand': False,
    'punctuation_remove': False, 'digits_remove': True,
    'negation_mark': True, 'normalize': True, 'stemming': True,
    'stopwords_remove': False, 'lower': False
},
{
    'metadata_remove': False, 'emoji_remove': False, 'negation_expand': False,
    'punctuation_remove': False, 'digits_remove': False,
    'negation_mark': False, 'normalize': False, 'stemming': False,
    'stopwords_remove': False, 'lower': False
}]
    model_params = [{'C': 0.001, 'penalty': 'l2'},
    {'criterion': 'gini', 'max_features': 'sqrt'},
    {'alpha': 0.01, 'penalty': 'l2'},
    {'alpha': 1.0}]

    vectorizer_params = [{'max_features': 3200, 'ngram_range': (1, 2)},
    {'max_features': 3200, 'ngram_range': (1, 1)},
    {'max_features': 3200, 'ngram_range': (1, 1)},
    {'max_features': 3200, 'ngram_range': (1, 1)}]

    models = [LogisticRegression(**model_params[0]),
    RandomForestClassifier(**model_params[1]),
    CalibratedClassifierCV(base_estimator=SGDClassifier(**model_params[2], max_iter=250)),
    MultinomialNB(**model_params[3])]

    dataframe = pd.read_table(file)
    col = ['tweetID', 'text', 'relevant']
    df = dataframe[col]
    print(df.info())
    x = df.text
    y = df.relevant

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42, stratify=y)
    results = pd.DataFrame(columns=['preprocess', 'vectorizer', 'algorithm', 'target', 'precision', 'recall', 'f1', 'accuracy', 'auroc',
                                    'auprc', 'brier', 'youden', 'precision_J', 'recall_J',
                                    'f1_J', 'accuracy_J'])
    auc_df = pd.DataFrame(columns=['classifiers', 'fpr', 'tpr', 'auc'])
    prc_df = pd.DataFrame(columns=['classifiers', 'precision', 'recall'])
    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot()
    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    for i in range(0, 4):
        nlp_args = nlp_params[i]
        pipe = Pipeline(
        steps=[
            ('preprocess', NLP_transformer(**nlp_args)),
            ('vectorizer', CountVectorizer(**vectorizer_params[i], lowercase=False, stop_words=None)),
            ('clf', models[i])
        ]
        )
        df, auc, prc, preds = evaluate_model(pipe, 'relevant', nlp_args, X_train, X_test, y_train, y_test)
        preds.to_csv(pipe.named_steps['clf'].__class__.__name__ + '_relevant_preds.csv')
        results = results.append(df)
        auc_df = auc_df.append(auc, ignore_index=True)
        prc_df = prc_df.append(prc, ignore_index=True)

        fraction_of_positives, mean_predicted_value, y_prob = calibration_plot(pipe, X_train, X_test, y_train, y_test)
        ax.plot(mean_predicted_value, fraction_of_positives, "s-",
                label="%s" % (pipe.named_steps['clf'].__class__.__name__ ))

    ax.legend(loc="lower right")
    ax.set_ylabel("Fraction of positives")
    ax.set_xlabel("Mean predicted value")
    ax.set_ylim([-0.05, 1.05])
    ax.set_title('Calibration plots (relevance ML classifiers)')
    plt.tight_layout()
    plt.savefig('results/calibration_plot_relevance_ML.png')

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

    plt.title('ROC Curve Analysis (relevance label)', fontweight='bold', fontsize=15)
    plt.legend(prop={'size':13}, loc='lower right')

    plt.savefig('results/relevance_AUC_ML_models.png')

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

    plt.title('PRC Curve Analysis (relevance label)', fontweight='bold', fontsize=15)
    plt.legend(prop={'size':13}, loc='lower right')
    plt.savefig('results/relevance_PRC_ML_models.png')

    file = dir_in+'data/D2.tsv'
    nlp_params = [{
        'metadata_remove': False, 'emoji_remove': False, 'negation_expand': False,
        'punctuation_remove': False, 'digits_remove': False,
        'negation_mark': False, 'normalize': False, 'stemming': False,
        'stopwords_remove': False, 'lower': False
    }, {
        'metadata_remove': False, 'emoji_remove': False, 'negation_expand': True,
        'punctuation_remove': True, 'digits_remove': True,
        'negation_mark': True, 'normalize': False, 'stemming': True, 'stopwords_remove': True,
        'lower': False
    },
    {
        'metadata_remove': False, 'emoji_remove': False, 'negation_expand': True,
        'punctuation_remove': True, 'digits_remove': True,
        'negation_mark': True, 'normalize': False, 'stemming': False,
        'stopwords_remove': True, 'lower': False
    },
    {
        'metadata_remove': False, 'emoji_remove': False, 'negation_expand': False,
        'punctuation_remove': False, 'digits_remove': False,
        'negation_mark': False, 'normalize': False, 'stemming': False,
        'stopwords_remove': False, 'lower': False
    }]
    model_params = [{'C': 1.0, 'penalty': 'l2'},
        {'criterion': 'gini', 'max_features': 'sqrt'},
        {'alpha': 0.0001, 'penalty': 'l2'},
        {'alpha': 0.01}]

    vectorizer_params = [{'max_features': 3200, 'ngram_range': (1, 1)},
        {'max_features': 1600, 'ngram_range': (1, 1)},
        {'max_features': 3200, 'ngram_range': (1, 1)},
        {'max_features': 3200, 'ngram_range': (1, 1)}]

    models = [LogisticRegression(**model_params[0]),
        RandomForestClassifier(**model_params[1]),
        CalibratedClassifierCV(base_estimator=SGDClassifier(**model_params[2], max_iter=250)),
        MultinomialNB(**model_params[3])]

    dataframe = pd.read_table(file)
    col = ['tweetID', 'text', 'com_vape']
    df = dataframe[col]
    print(df.info())
    x = df.text
    y = df.com_vape

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42, stratify=y)
    auc_df = pd.DataFrame(columns=['classifiers', 'fpr', 'tpr', 'auc'])
    prc_df = pd.DataFrame(columns=['classifiers', 'precision', 'recall'])

    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot()
    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    for i in range(0, 4):
        nlp_args = nlp_params[i]
        pipe = Pipeline(
        steps=[
            ('preprocess', NLP_transformer(**nlp_args)),
            ('vectorizer', CountVectorizer(**vectorizer_params[i], lowercase=False, stop_words=None)),
            ('clf', models[i])
        ]
        )
        df, auc, prc, preds = evaluate_model(pipe, 'com_vape', nlp_args, X_train, X_test, y_train, y_test)
        preds.to_csv(pipe.named_steps['clf'].__class__.__name__ + '_com_vape_preds.csv')
        results = results.append(df)
        auc_df = auc_df.append(auc, ignore_index=True)
        prc_df = prc_df.append(prc, ignore_index=True)

        fraction_of_positives, mean_predicted_value, y_prob = calibration_plot(pipe, X_train, X_test, y_train, y_test)
        ax.plot(mean_predicted_value, fraction_of_positives, "s-",
                label="%s" % (pipe.named_steps['clf'].__class__.__name__ ))

    ax.legend(loc="lower right")
    ax.set_ylabel("Fraction of positives")
    ax.set_xlabel("Mean predicted value")
    ax.set_ylim([-0.05, 1.05])
    ax.set_title('Calibration plots (commercial ML classifiers)')
    plt.tight_layout()
    plt.savefig('results/calibration_plot_comvape_ML.png')

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

    plt.title('ROC Curve Analysis (commerical label)', fontweight='bold', fontsize=15)
    plt.legend(prop={'size':13}, loc='lower right')

    plt.savefig('results/commercial_AUC_ML_models.png')

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

    plt.title('PRC Curve Analysis (commercial label)', fontweight='bold', fontsize=15)
    plt.legend(prop={'size':13}, loc='lower right')
    plt.savefig('results/commercial_PRC_ML_models.png')

    file = dir_in+'data/D3.tsv'
    nlp_params = [{
        'metadata_remove': False, 'emoji_remove': False, 'negation_expand': False,
        'punctuation_remove': False, 'digits_remove': False, 'negation_expand': False,
        'negation_mark': False, 'normalize': False, 'stemming': False,
        'stopwords_remove': False, 'lower': False
    }, {
        'metadata_remove': True, 'emoji_remove': False, 'negation_expand': False,
        'punctuation_remove': False, 'digits_remove': True, 'negation_expand': True,
        'negation_mark': False, 'normalize': False, 'stemming': True, 'stopwords_remove': False,
        'lower': False
    },
    {
        'metadata_remove': False, 'emoji_remove': True, 'negation_expand': False,
        'punctuation_remove': True, 'digits_remove': True,
        'negation_mark': False, 'normalize': True, 'stemming': True,
        'stopwords_remove': True, 'lower': True
    },
    {
        'metadata_remove': False, 'emoji_remove': False, 'negation_expand': False,
        'punctuation_remove': False, 'digits_remove': False,
        'negation_mark': False, 'normalize': False, 'stemming': False,
        'stopwords_remove': False, 'lower': False
    }]
    model_params = [{'C': 0.1, 'penalty': 'l2'},
        {'criterion': 'gini', 'max_features': 'sqrt'},
        {'alpha': 0.0001, 'penalty': 'l2'},
        {'alpha': 1.0}]

    vectorizer_params = [{'max_features': 3200, 'ngram_range': (1, 2)},
        {'max_features': 800, 'ngram_range': (1, 1)},
        {'max_features': 3200, 'ngram_range': (1, 1)},
        {'max_features': 3200, 'ngram_range': (1, 2)}]

    models = [LogisticRegression(**model_params[0]),
        RandomForestClassifier(**model_params[1]),
        CalibratedClassifierCV(base_estimator=SGDClassifier(**model_params[2], max_iter=250)),
        MultinomialNB(**model_params[3])]

    dataframe = pd.read_table(file)
    col = ['tweetID', 'text', 'pro_vape']
    df = dataframe[col]
    print(df.info())
    x = df.text
    y = df.pro_vape

    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot()
    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42, stratify=y)
    auc_df = pd.DataFrame(columns=['classifiers', 'fpr', 'tpr', 'auc'])
    prc_df = pd.DataFrame(columns=['classifiers', 'precision', 'recall'])

    for i in range(0, 4):
        nlp_args = nlp_params[i]
        pipe = Pipeline(
        steps=[
            ('preprocess', NLP_transformer(**nlp_args)),
            ('vectorizer', CountVectorizer(**vectorizer_params[i], lowercase=False, stop_words=None)),
            ('clf', models[i])
        ]
        )
        df, auc, prc, preds = evaluate_model(pipe, 'pro_vape', nlp_args, X_train, X_test, y_train, y_test)
        results = results.append(df)
        preds.to_csv(pipe.named_steps['clf'].__class__.__name__ + '_pro_vape_preds.csv')
        auc_df = auc_df.append(auc, ignore_index=True)
        prc_df = prc_df.append(prc, ignore_index=True)

        fraction_of_positives, mean_predicted_value, y_prob = calibration_plot(pipe, X_train, X_test, y_train, y_test)
        ax.plot(mean_predicted_value, fraction_of_positives, "s-",
                label="%s" % (get_label(pipe) ))

    ax.legend(loc="lower right")
    ax.set_ylabel("Fraction of positives")
    ax.set_xlabel("Mean predicted value")
    ax.set_ylim([-0.05, 1.05])
    ax.set_title('Calibration plots (pro-vape sentiment ML classifiers)')
    plt.tight_layout()
    plt.savefig('results/calibration_plot_provape_ML.png')

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

    plt.title('PRC Curve Analysis (pro-vape label)', fontweight='bold', fontsize=15)
    plt.legend(prop={'size':13}, loc='lower right')
    plt.savefig('results/provape_PRC_ML_models.png')

    results.to_csv('ML_results.csv', index=False)
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

    plt.title('ROC Curve Analysis (pro-vape label)', fontweight='bold', fontsize=15)
    plt.legend(prop={'size':13}, loc='lower right')

    plt.savefig('results/provape_AUC_ML.png')
