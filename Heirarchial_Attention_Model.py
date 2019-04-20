import numpy
import numpy as np
from numpy import asarray
from numpy import zeros
import pandas as pd
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding
from keras.layers import LSTM, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.core import*
from keras import initializers, regularizers, constraints, Input
from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras import backend as K


import codecs
import csv
from nltk import word_tokenize


from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split


class Attention(Layer):
	def __init__(self,
				 W_regularizer=None, b_regularizer=None,
				 W_constraint=None, b_constraint=None,
				 bias=True, **kwargs):

		self.supports_masking = True
		self.init = initializers.get('glorot_uniform')

		self.W_regularizer = regularizers.get(W_regularizer)
		self.b_regularizer = regularizers.get(b_regularizer)

		self.W_constraint = constraints.get(W_constraint)
		self.b_constraint = constraints.get(b_constraint)

		self.bias = bias
		super(Attention, self).__init__(**kwargs)

	def build(self, input_shape):
		assert len(input_shape) == 3

		self.W = self.add_weight((input_shape[-1],),
								 initializer=self.init,
								 name='{}_W'.format(self.name),
								 regularizer=self.W_regularizer,
								 constraint=self.W_constraint)
		if self.bias:
			self.b = self.add_weight((input_shape[1],),
									 initializer='zero',
									 name='{}_b'.format(self.name),
									 regularizer=self.b_regularizer,
									 constraint=self.b_constraint)
		else:
			self.b = None

		self.built = True

	def compute_mask(self, input, input_mask=None):
		return None

	def call(self, x, mask=None):
		eij = K.squeeze(K.dot(x, K.expand_dims(self.W)), axis=-1)

		if self.bias:
			eij += self.b

		eij = K.tanh(eij)

		a = K.exp(eij)

		if mask is not None:
			a *= K.cast(mask, K.floatx())

		a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

		a = K.expand_dims(a)

		weighted_input = x * a
		return K.sum(weighted_input, axis=1)

	def compute_output_shape(self, input_shape):
		return (input_shape[0], input_shape[-1])

def precision(y_true, y_pred):

	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
	precision = true_positives / (predicted_positives + K.epsilon())
	return precision


def recall(y_true, y_pred):

	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
	recall = true_positives / (possible_positives + K.epsilon())
	return recall


def f1(y_true, y_pred):
	def recall(y_true, y_pred):

		true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
		possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
		recall = true_positives / (possible_positives + K.epsilon())
		return recall

	def precision(y_true, y_pred):

		true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
		predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
		precision = true_positives / (predicted_positives + K.epsilon())
		return precision
	precision = precision(y_true, y_pred)
	recall = recall(y_true, y_pred)
	return 2*((precision*recall)/(precision+recall+K.epsilon()))



def ReadFile(filename):
	with open(filename) as tsvfile:
		reader = csv.reader(tsvfile, delimiter='\t')
		comments = []
		labels = []
		for row in reader:
			comments.append(row[2])
			labels.append(row[1])

	return comments[:],labels[:]

def CalCount(comments):
	total = 0
	for row in comments:
		total += len(row)
	return total


def Preprocessing(docs,count):
	t = Tokenizer()
	t.fit_on_texts(docs)
	# integer encode the documents
	encoded_docs = t.texts_to_sequences(docs)
	# pad documents to a max length of 4 words
	# max_length = 4
	padded_docs = pad_sequences(encoded_docs, padding='post')
	l = len(padded_docs[0])

	# load the whole embedding into memory
	embeddings_index = dict()
	f = open('glove.6B.100d.txt')
	for line in f:
		values = line.split()
		word = values[0]
		coefs = asarray(values[1:], dtype='float32')
		embeddings_index[word] = coefs
	f.close()
	print('Loaded %s word vectors.' % len(embeddings_index))

	# create a weight matrix for words in training docs
	embedding_matrix = zeros((count, 100))
	for word, i in t.word_index.items():
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector

	return padded_docs,embedding_matrix,l



def PrepModel(count,embedding_matrix,l,lrate=0.001):
	model = Sequential()
	e = Embedding(count, 100, weights=[embedding_matrix], input_length=l, trainable=False)
	model.add(e)

	model.add(LSTM(100,kernel_initializer='he_normal', activation='sigmoid', dropout=0.5,recurrent_dropout=0.5, unroll=False, return_sequences=True))

	model.add(Attention())
	model.add(Dense(1, activation='sigmoid'))
	# model.compile(optimizer=Adam(lr=lrate), loss='binary_crossentropy', metrics=["accuracy"])
	model.compile(optimizer="rmsprop", loss='binary_crossentropy', metrics=["acc",f1,precision,recall])


	print('No of parameter:', model.count_params())

	print(model.summary())
	print("Learning rate:",K.eval(model.optimizer.lr))
	return model


if __name__ == "__main__":
	seed = 7
	numpy.random.seed(seed)
	num_epochs=50
	filename = "dEFEND data/gossipcop_content_no_ignore.tsv"
	print("Reading data...")
	comments,labels = ReadFile(filename)
	labels = numpy.array(labels)

	print("Getting word count...")
	word_count = CalCount(comments)
	print("Number of words:",word_count)

	print("Preprocessing...")
	padded_docs,embedding_matrix,l = Preprocessing(comments,word_count)

	print("Preparing model...")
	model = PrepModel(word_count,embedding_matrix,l)

	print('Training...')

	X_train, X_test, y_train, y_test = train_test_split(padded_docs, labels, test_size=0.2, random_state=seed)
	X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=seed)
	print("length of X_test,y_test")
	print(len(X_test),len(y_test))

	# earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=0, verbose=1, mode='auto')
	model.fit(X_train, y_train, validation_data=(X_val,y_val), verbose=1,nb_epoch=num_epochs,shuffle=True)
	loss, accuracy,f1_score,precision,recall = model.evaluate(X_test, y_test, verbose=1)
	print('Accuracy: %f' % (accuracy*100))
