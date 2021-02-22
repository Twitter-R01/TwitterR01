import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gensim
import multiprocessing
import tensorflow as tf
from gensim.models import KeyedVectors
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn import utils
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Conv1D, Dense, Dropout, Flatten, GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.callbacks import Callback, EarlyStopping


# function to generate neural networks for cross validation
def create_network(optimizer='rmsprop', filters=25, kernel_size=1):
    # initiate convolutional neural network
    network = Sequential()

    # load embeddings
    e = Embedding(2000, 200, weights=[embedding_matrix], input_length=60, trainable=False)
    network.add(e)

    # add Conv1D layer
    network.add(Conv1D(filters=filters, kernel_size=kernel_size, padding='valid', activation='relu', strides=1))

    # Global Max Pooling layer
    network.add(GlobalMaxPooling1D())

    # Add hidden layer
    network.add(Dense(256, activation='relu'))

    # Add classification layer
    network.add(Dense(1, activation='sigmoid'))

    # Compile CNN
    network.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', auc_roc])
    return network


def auc_roc(y_true, y_pred):
    # any tensorflow metric
    value, update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true)

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value


# read in data, drop na rows
d1 = pd.read_table('./data/processed_D1.tsv')
d1.dropna(inplace=True)
d1.reset_index(drop=True, inplace=True)
x = d1.clean_text
y = d1.relevant
# train/test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42, stratify=y)

# load pre-trained W2V models
model_ug_cbow = KeyedVectors.load('w2v_model_ug_cbow.word2vec')
model_ug_sg = KeyedVectors.load('w2v_model_ug_sg.word2vec')

# build embeddings
embeddings_index = {}
for w in model_ug_cbow.wv.vocab.keys():
    embeddings_index[w] = np.append(model_ug_cbow.wv[w],model_ug_sg.wv[w])
print('Found %s word vectors.' % len(embeddings_index))

tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(x_train)
sequences = tokenizer.texts_to_sequences(x_train)

x_train_seq = pad_sequences(sequences, maxlen=60)
sequences_test = tokenizer.texts_to_sequences(x_test)
x_test_seq = pad_sequences(sequences_test, maxlen=60)

num_words = 2000
embedding_matrix = np.zeros((num_words, 200))
for word, i in tokenizer.word_index.items():
    if i >= num_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Create model parameter space
epochs = [5, 10]
batches = [16, 32, 64]
optimizers = ['rmsprop', 'adam']
filters = [25, 50, 100]
kernel_sizes = [1, 2, 3]

# Wrap Keras model for sklearn GridSearchCV
conv_neural_network = KerasClassifier(build_fn=create_network, verbose=0)

params = dict(optimizer=optimizers, filters=filters, kernel_size=kernel_sizes,
              epochs=epochs, batch_size=batches)

# Create grid search
grid = GridSearchCV(estimator=conv_neural_network, param_grid=params)

# Fit grid search
grid_result = grid.fit(x_train_seq, y_train)

print('Best estimator: ', grid_result.best_estimator_)
print('Best score: ', grid_result.best_score_)
print('Best parameters: ', grid_result.best_params_)

y_pred = grid_result.predict(x_test_seq)
test_score = grid_result.score(x_test_seq, y_test)
print('Test Score: ', test_score)
