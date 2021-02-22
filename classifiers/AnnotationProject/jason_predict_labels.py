#import multiprocessing
import sys
import pandas as pd
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences


#import importlib
''' Alycia Edited '''
sys.path.append('/home/ancarey/RITHM/parser/') # path to folder with parselogic.py
import parselogic


def classify(model, tweets):
        y_pred = model.predict(tweets)
        return y_pred


if __name__ == '__main__':

    #load models
    #model_path = '/pylon5/be5fpap/jcolditz/scripts/apply_LSTM/'
    ''' Alycia Edited '''
    model_path = '/home/ancarey/AnnotationProjects/classification/'
    lstm_relevance = load_model(model_path+'lstm_model_relevance_new.h5')
    lstm_commercial = load_model(model_path+'lstm_model_commercial_new.h5')
    lstm_pro_vape = load_model(model_path+'lstm_model_pro_vape_new.h5')

    ''' Alycia Edited '''
    with open('/home/ancarey/AnnotationProjects/classification/lstm_tokenizer_new.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    #other defaults
    #dir_path = '/pylon5/be5fpap/jcolditz/parser_out/db_data/LSTM_test/'
    ''' Alycia Edited '''
    #dir_path = '/pylon5/be5fpap/jcolditz/parser_out/juul_fda/juul_classification_20180301-20181231/'
    maxlen = 75
    start = 0
    end = 99999999

    #for saving counts of tweets after classification - can edit
    save = True
    threshold = 'custom'
    retweet_ignore = True
    retweet_status = '_noRTs'
    fname_summary = 'summary'
    
    # sets classifier threshold to use (custom is from Visweswaran et al. 2020)
    if threshold == 'default':
        threshold_values = [0.5, 0.5, 0.5]
    else:
        threshold_values = [0.847, 0.344, 0.837]

    count_dict = {'date': [],
    'tweets': [],
    'relevant': [],
    'commercial': [],
    'non_commercial': [],
    'pro_vape': [],
    'not_pro': []}


    #get command line arguments
    cv = parselogic.cmdvars()
    start = 0 #cv['start']
    end = 99999999 #cv['end']
    # dir_in = cv['dir_in']
    #dir_in = '/pylon5/be5fpap/jcolditz/parser_out/db_data/LSTM_test/'
    #dir_in = '/pylon5/be5fpap/jcolditz/parser_out/juul_fda/juul_classification_20180301-20181231/'
    dir_in = '/home/ancarey/AnnotationProjects/classification/vape_long_LSTM_test_data/'
    dir_out = '/home/ancarey/AnnotationProjects/classification/vape_long_classification/' 
    #dir_out = cv['dir_out']
    f1 = cv['f1']
    f2 = cv['f2']
    f_ext = cv['f_ext']
    delimiter = cv['delimiter']
    rt_include = cv['rt_include']
    rt_ignore = cv['rt_ignore']
    rt_status = cv['rt_status']
    f_stem = cv['f_stem']


    datafiles = parselogic.filelist(dir_in, start=start, end=end)

    for file in datafiles:
        df = pd.read_table(dir_in+file)

        #find and remove duplicate tweet IDs before running
        df = df.drop_duplicates()

        #optionlly remove RTs before applying classifiers to the data (default is to remove)
        if rt_ignore:
            df = df[pd.isna(df['rt_t_tid'])]

        #combine text and quote for prediction - removed (this is not how the classifier was trained)
        #texts = df['t_text'].str.cat(df['t_quote'], sep=' ', na_rep='')
        texts = df['t_text']

        #tokenize and format tweets for prediction
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

        
        #TO-DO: optionally reduce tweet-level output to only t_id and classification values for database efficiency
        
        print('Saving results of file: ', str(file))
        #Update the following line after there is a process to create output directory
        results.to_csv(dir_out+str(file[:14])+rt_status+f_ext, sep='\t', index=False)
        

        #save counts to dictionary
        if save:
            count_dict['date'].append(int(file[:8]))
            count_dict['tweets'].append(len(results.index))

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

    if save:
        count_df = pd.DataFrame(data=count_dict)
        count_df = count_df.groupby(['date']).sum() 
        print('\nSaving counts to file')
        #Update the following line after there is a process to create output directory
        #count_df.to_csv(dir_out+'MLsummary'+rt_status+f_stem+f_ext, sep=delimiter, columns = ['tweets', 'relevant', 'commercial', 'non_commercial', 'pro_vape', 'not_pro'])
        count_df.to_csv(dir_out+'MLsummary', sep=delimiter, columns = ['tweets', 'relevant', 'commercial', 'non_commercial', 'pro_vape', 'not_pro'])

    print('\nProcess complete!\n\n')