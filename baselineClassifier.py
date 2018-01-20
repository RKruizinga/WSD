import numpy as np

import sklearn
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from nltk.tokenize import TweetTokenizer

from basicFunctions import BasicFunctions

class Baseline:
  X_train = []
  Y_train = []
  X_test = []
  Y_test = []

  Y_predicted = []
  labels = []

  def __init__(self, X_train, X_test, Y_train, Y_test, labels):
    self.X_train = X_train
    self.X_test = X_test
    self.Y_train = Y_train
    self.Y_test = Y_test

    self.labels = labels

    self.classifier = Classifier()

  def classify(self):
    self.classifier.fit(self.X_train, self.Y_train)  

  def evaluate(self):
    self.Y_predicted = self.classifier.predict(self.X_test)

    self.accuracy, self.precision, self.recall, self.f1score = BasicFunctions.getMetrics(self.Y_test, self.Y_predicted, self.labels)

  def printBasicEvaluation(self):    
    BasicFunctions.printEvaluation(self.accuracy, self.precision, self.recall, self.f1score, "Basic Evaluation")

  def printClassEvaluation(self):
   BasicFunctions.printClassEvaluation(self.Y_test, self.Y_predicted, self.labels)

class Classifier:
  def __init__(self):
    pass

  def fit(self, X, y):
    label_distribution = BasicFunctions.keyCounter(y)
    highest_amount = 0
    for label in label_distribution:
      if label_distribution[label] > highest_amount or highest_amount == 0:
        highest_amount = label_distribution[label]
        self.most_frequent_class = label
  
  def predict(self, X):
    Y_predicted = []
    for doc in X:
      Y_predicted.append(self.most_frequent_class) 
    return Y_predicted


