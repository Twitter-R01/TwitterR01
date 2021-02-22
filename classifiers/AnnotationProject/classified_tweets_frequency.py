import pandas as pd
import sys, os
import pickle
import numpy as np
import re

'''
if retweet_ignore is True, return dataframe without retweets
'''
def get_original_tweets(data):
	results = data[pd.isna(data['rt_t_tid'])]
	return results

if __name__ == '__main__':
	
	#load files
	dir_path = '/pylon5/be5fpap/jcolditz/parser_out/juul_fda/juul_classification_20180301-20181231/classification_output/'
	start = None
	end = None
	retweet_ignore = True
	if len(sys.argv) > 1: # If command line arguments were passed
		i = 0
		for arg in sys.argv:
			# '-d' indicates start/end dates (req. 2 objects)
			if arg.lower() in ['-d','-date','-dates']:
				start = int(sys.argv[i+2])
				end = int(sys.argv[i+3])

	files = os.listdir(dir_path)
	files.sort()
	print('\n\nSTART_DATE: '+str(start))
	print('END_DATE:   '+str(end))

	threshold = 'custom'
	
	if threshold == 'default':
		threshold_values = [0.5, 0.5, 0.5]
	else:
		threshold_values = [0.847, 0.344, 0.837]

	count_dict = {'date': [],
	'tweets': [],
	'original_tweets': [],
	'relevant': [],
	'commercial': [],
	'non_commercial': [],
	'pro_vape': [],
	'not_pro': []}

	for file in files:
		if file[-4:] == '.tsv':
			if int(file[:8]) >= int(start):
				if int(file[:8]) <=int(end):
					results = pd.read_csv(dir_path+file, delimiter='\t')

					count_dict['date'].append(int(file[:8]))
					count_dict['tweets'].append(len(results.index))
					if retweet_ignore:
						results = get_original_tweets(results)
						count_dict['original_tweets'].append(len(results.index))
					
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
	print('Saving counts to file')
	count_df = pd.DataFrame(data=count_dict)
	count_df = count_df.groupby(['date']).sum()	
	count_df.to_csv('/pylon5/be5fpap/jcolditz/parser_out/juul_fda/juul_classification_20180301-20181231/classification_output/tweet_frequency_custom_2018_no_retweets.tsv', sep='\t', columns = ['tweets', 'original_tweets', 'relevant', 'commercial', 'non_commercial', 'pro_vape', 'not_pro'])





