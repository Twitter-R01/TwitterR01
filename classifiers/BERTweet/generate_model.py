# Authors: William Baker, 
# Purpose: This script uses the SimpleTransformers API along with BERTweet, a transformer deep learning model, to
#          generate and train one model per execution of this script.
# Usage:   Example execution: > python(3) generate_model.py relevance D1.tsv data
#          This code has not been tested in any other version of Python besides Python 3, hence the optional (3) in 
#          the execution. The first argument is the target variable of the classifier you wish to create.
#          The second argument is the .tsv file which holds the tweet IDs and codings for each tweet. The third argument is 
#          the directory in which the parsed tweet files are held. These parsed files should span the exact date range that
#          the coded tweets were collected in.

import sys
import logging
import pandas as pd
import numpy as np
import os
from simpletransformers.classification import ClassificationModel
from sklearn.model_selection import train_test_split
from torch.cuda import is_available
from sklearn.metrics import roc_auc_score
from scipy.special import softmax

def classification_model(num_labels, output_dir):
    return ClassificationModel('bertweet', 'vinai/bertweet-base', num_labels=num_labels, use_cuda=is_available(),
                                      args={'num_train_epochs': 10, 
                                            'train_batch_size': 32,
                                            'output_dir': output_dir,
                                            'overwrite_output_dir': True})

def read_coded_file(tweet_codes_file, column_names):
    df_annotated = pd.read_csv(tweet_codes_file, header=0, usecols=['t_id']+column_names, index_col='t_id')
    df_annotated.dropna(subset=column_names, inplace=True)
    id_list = df_annotated.index.tolist()
    return df_annotated, id_list

def generate_final_df(df_annotated, id_list, parsed_directory):
    final_df = pd.DataFrame(columns=['t_id','t_text','t_quote'], dtype='str')
    for subdir, dirs, files in os.walk(parsed_directory):
        for filename in files:
            filepath = subdir + os.sep + filename
            if filepath.endswith(".tsv"):
                df = pd.read_csv(filepath, sep='\t', 
                                 usecols=['t_id','t_text','t_quote'],
                                 dtype='str')
                final_df = final_df.append(df[df['t_id'].isin(id_list)])
    final_df.t_quote.replace(np.nan, '', inplace=True, regex=True)
    final_df['text'] = final_df.t_text + ' ' + final_df.t_quote
    final_df = final_df.drop(columns=['t_text','t_quote']).set_index('t_id')
    final_df = final_df.join(df_annotated, how='inner')
    missing_tweet_count = df_annotated.index.difference(final_df.index).shape[0]
    if missing_tweet_count != 0:
        logging.warning(f'Could not retrieve {missing_tweet_count} coded tweets from parsed data files')
    return final_df
    
def split_data(df):
    X_train, X_test, y_train, y_test = train_test_split(df.index.values, 
                                                        df.labels.values, 
                                                        test_size=0.1,
                                                        stratify=df.labels.values)
    df.loc[X_train,'data_type'] = 'train'
    df.loc[X_test,'data_type'] = 'test'
    train_df = df[df['data_type']=='train'].drop(columns='data_type')
    test_df = df[df['data_type']=='test'].drop(columns='data_type')
    return train_df, test_df
    
def evaluate_model(model, test_df):
    result, model_outputs, wrong_predictions = model.eval_model(test_df)
    print(result)

def main():
    classifier_type = sys.argv[1]
    tweet_codes_file = sys.argv[2]
    parsed_directory = sys.argv[3]
        
    if classifier_type == 'relevance':
        model = classification_model(2, 'relevance/')
        column_names = ['Relevant_1']
        df_annotated, id_list = read_coded_file(tweet_codes_file, column_names)
    elif classifier_type == 'commercial':
        model = classification_model(2, 'commercial/')
        column_names = ['Com.']
        df_annotated, id_list = read_coded_file(tweet_codes_file, column_names)
    elif classifier_type == 'policy':
        model = classification_model(3, 'policy/')
        column_names = ['Pro Policy', 'Anti Policy', 'Neither Policy']
        df_annotated, id_list = read_coded_file(tweet_codes_file, column_names)
    elif classifier_type == 'sentiment':
        model = classification_model(4, 'sentiment/')
        column_names = ['Pro Vape Sent.', 'Anti Vape Sent. ', 'Neither Sent.']
        df_annotated, id_list = read_coded_file(tweet_codes_file, column_names)
    else:
        logging.error('First argument must be one of the following: "relevance", "commercial", "policy", or "sentiment"')
        return

    final_df = generate_final_df(df_annotated, id_list, parsed_directory)
    if classifier_type == 'sentiment':
        final_df['no_sentiment'] = final_df[column_names].max(axis=1).replace({0.0:1.0,1.0:0.0})
        column_names.append('no_sentiment')
        final_df['labels'] = final_df[column_names].idxmax(axis=1).apply(
            lambda x: (column_names).index(x))
    elif classifier_type == 'policy':
        final_df['labels'] = final_df[column_names].idxmax(axis=1).apply(
            lambda x: (column_names).index(x))
    else:
        final_df['labels'] = final_df[column_names]
        
    df = final_df.drop(columns=column_names)
    train_df, test_df = split_data(df)
    model.train_model(train_df)
    
    evaluate_model(model, test_df)
    predictions, raw_outputs = model.predict(test_df.text.tolist())
    if classifier_type == 'policy' or classifier_type == 'sentiment':
        roc_score = roc_auc_score(test_df.labels.tolist(), softmax(raw_outputs, axis=1), multi_class='ovo')
        print(f'ROC: {roc_score}')

    return
	
	
if __name__ == '__main__':
	main()
