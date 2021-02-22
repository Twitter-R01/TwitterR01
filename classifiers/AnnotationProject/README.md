## Classification of Twitter data

### Levels of classification
1. Relevance
2. Commercial
3. Pro_vape

### Data
The /data folder contains the TSV files used for training and testing of the classifiers. Details of the data files can be found [here](https://github.com/CRMTH/AnnotationProjects/blob/master/classification/data/README.md).
The script [create_dataset](https://github.com/CRMTH/AnnotationProjects/blob/master/classification/create_dataset.py) is used to divide the data into separate files for different categories.

### Training the classifiers
The scripts [evaluate_models](https://github.com/CRMTH/AnnotationProjects/blob/master/classification/evaluate_models.py) and [evaluate_deep_learning](https://github.com/CRMTH/AnnotationProjects/blob/master/classification/evaluate_deep_learning.py) are used to train the ML and DL models and get the performance of the classifiers. Classifiers include:
- Logistic Regression
- Random Forest
- Naive Bayes
- SVM
- CNN
- LSTM
- CNN-LSTM
- biLSTM

Helper scripts for classification: [nlp_preprocess](https://github.com/CRMTH/AnnotationProjects/blob/master/classification/nlp_preprocess.py) for NLP pipeline, embedding_tweets.ipynb to create vape embeddings from data

### Using the LSTM classifier

1. [train_LSTM_models](https://github.com/CRMTH/AnnotationProjects/blob/master/classification/train_LSTM_models.py) is used to train and save the models for all 3 categories and the tokenizer. The saved models and tokenizer can be found in this repository (lstm_tokenizer.pickle, lstm_model_commercial.h5, lstm_model_pro_vape.h5, lstm_model_relevance.h5).
2. [predict_labels_LSTM](https://github.com/CRMTH/AnnotationProjects/blob/master/classification/predict_labels_LSTM.py) is then used to classify new data from TSV files using the saved models and outputs TSV files with the tweets and the probabilities of classification. It further gives a per-day frequency of the classified tweets.
3. If the classified data already exists, [classified_tweets_frequency](https://github.com/CRMTH/AnnotationProjects/blob/master/classification/classified_tweets_frequency.py) can be used to find the per-day frequency of the classified tweets for all categories.
