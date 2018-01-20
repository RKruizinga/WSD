import numpy as np
import re

from sklearn.model_selection import train_test_split

from basicFunctions import BasicFunctions
from customTokenizer import CustomTokenizer

import keras
import keras.backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import Embedding
from keras.layers import LSTM

class NeuralNetwork:
  labels = []
  labels_dict = {}
  labels_dict_rev = {}

  avoid_skewness = False
  
  Y = []

  def __init__(self, X, Y, labels, avoid_skewness):
    self.X = X

    self.avoid_skewness = avoid_skewness

    self.labels = labels
    for i, label in enumerate(self.labels):
      self.labels_dict[label] = i
      self.labels_dict_rev[i] = label

    for label in Y:
      self.Y.append(self.labels_dict[label])

  def tokenize(self):
    self.X_tokenized = CustomTokenizer.tokenizeTweets(self.X) #all tweets!
    self.tokenizer = Tokenizer(split="|",)
    self.tokenizer.fit_on_texts(self.X_tokenized)
    self.sequences = self.tokenizer.texts_to_sequences(self.X_tokenized)
    self.X = pad_sequences(self.sequences)
    self.Y = to_categorical(self.Y)

  def classify(self):
    self.tokenize()
    self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y,  test_size=0.20, random_state=42)
    if self.avoid_skewness:
      Y_train = np.argmax(self.Y_train, axis=1)
      Y_train = [self.labels_dict_rev[int(i)] for i in list(Y_train)]
      
      self.X_train, self.Y_train = BasicFunctions.getUnskewedSubset(self.X_train, self.Y_train, Y_train)
      self.X_train = np.array(self.X_train)
      self.Y_train = np.array(self.Y_train)
    self.printDataInformation()

    self.model = Sequential()
    # Single 500-neuron hidden layer with sigmoid activation
    self.model.add(Dense(input_dim = self.X.shape[1], units = 500, activation = 'relu'))
    # Output layer with softmax activation
    self.model.add(Dense(units = self.Y.shape[1], activation = 'softmax'))
    # Specify optimizer, loss and validation metric
    self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	
    # Train the model 
    self.model.fit(self.X_train, self.Y_train, epochs = 4, batch_size = 10, validation_split = 0.2)

  def evaluate(self):
    self.Y_predicted = self.model.predict(self.X_test)
    self.Y_predicted = np.argmax(self.Y_predicted, axis=1)
    self.Y_predicted = [self.labels_dict_rev[int(i)] for i in list(self.Y_predicted)]
    
    self.Y_test = np.argmax(self.Y_test, axis=1)
    self.Y_test = [self.labels_dict_rev[int(i)] for i in list(self.Y_test)]

    self.accuracy, self.precision, self.recall, self.f1score = BasicFunctions.getMetrics(self.Y_test, self.Y_predicted, self.labels)

  def printBasicEvaluation(self):    
    BasicFunctions.printEvaluation(self.accuracy, self.precision, self.recall, self.f1score, "Basic Evaluation")

  def printClassEvaluation(self):
    BasicFunctions.printClassEvaluation(self.Y_test, self.Y_predicted, self.labels)

  def printDataInformation(self):

    print('~~~Neural Network Distribution~~~\n')
    print('Found {} unique tokens.'.format(len(self.tokenizer.word_index)))
    print('Shape of data tensor: {}'.format(self.X.shape))
    print('Shape of label tensor: {}\n'.format(self.Y.shape))

    
    


