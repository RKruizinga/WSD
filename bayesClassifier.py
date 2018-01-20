import numpy as np

import sklearn
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from basicFunctions import BasicFunctions
from customTokenizer import CustomTokenizer

class Bayes:
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

  def classify(self):
    self.classifier = Pipeline([('feats', FeatureUnion([
	 					 ('char', TfidfVectorizer(
                tokenizer=CustomTokenizer.wordTokenize, 
                lowercase=False, 
                analyzer='char', 
                binary=True, 
                ngram_range=(3,5), 
                min_df=1
              )),#, max_features=100000)),
	 					 ('word', TfidfVectorizer(
                tokenizer=CustomTokenizer.wordTokenize, 
                lowercase=False, 
                analyzer='word', 
                binary=True, 
                ngram_range=(1,5), 
                min_df=1
              )),#, max_features=100000)),
      ])),   
      ('classifier', MultinomialNB())
   ])

    self.classifier.fit(self.X_train, self.Y_train)  

  def evaluate(self):
    self.Y_predicted = self.classifier.predict(self.X_test)

    self.accuracy, self.precision, self.recall, self.f1score = BasicFunctions.getMetrics(self.Y_test, self.Y_predicted, self.labels)

  def printBasicEvaluation(self):    
    BasicFunctions.printEvaluation(self.accuracy, self.precision, self.recall, self.f1score, "Basic Evaluation")

  def printClassEvaluation(self):
   BasicFunctions.printClassEvaluation(self.Y_test, self.Y_predicted, self.labels)

