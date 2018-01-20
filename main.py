
import argparse
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from svmClassifier import SVM
from bayesClassifier import Bayes
from kNeighborsClassifier import KNeighbors
from decisionTreeClassifier import DecisionTree
from baselineClassifier import Baseline

from data import data

from basicFunctions import BasicFunctions

random.seed(3)

# Read arguments
parser = argparse.ArgumentParser(description='system parameters')
parser.add_argument('--method', type=str, default='svm', help='machine learning technique')
parser.add_argument('--data_method', type=int, default=1, help='how to divide the data') #all documents from 1 user in one string, or every document in one string
parser.add_argument('--predict_languages', type=str, default='e', help='predict languages: language name seperated with a comma or first letter of the language (without komma)') # ONLY USE English and Spanish for AGE!
parser.add_argument('--predict_label', type=str, default='Word Sense Disambiguation', help='word sense')
parser.add_argument('--avoid_skewness', type=bool, default=False, help='how to train the dataset, without skewness in the data or with skewness')
parser.add_argument('--kfold', type=int, default=1, help='Amount of Ks for cross validation, if cross validation.')
args = parser.parse_args()

predict_languages = BasicFunctions.getLanguages(args.predict_languages)

data = data('./train/', './test_00/')
data.collectXY(data_method=args.data_method) 

BasicFunctions.printStandardText(args.method, predict_languages, args.predict_label)
labels = list(set(data.Y))
#BasicFunctions.printLabelDistribution(data.Y_train)

words = list(set(data.words))
#print(data.X_train)
#BasicFunctions.printLabelDistribution(data.words_train)

if len(labels) > 1: #otherwise, there is nothing to train
 
  if args.method != 'neural':
    
    if args.avoid_skewness:
      X_train, Y_train = BasicFunctions.getUnskewedSubset(X_train, Y_train)

  if args.method == 'bayes':
    classifier = Bayes(data.X_train, data.X_test, data.Y_train, data.Y_test, labels) 
  elif args.method == 'svm':
    classifier = SVM(data.X_train, data.X_test, data.Y_train, data.Y_test, labels, words) 
  elif args.method == 'knear':
    classifier = KNeighbors(data.X_train, data.X_test, data.Y_train, data.Y_test, labels)
  elif args.method == 'tree':
    classifier = DecisionTree(data.X_train, data.X_test, data.Y_train, data.Y_test, labels)
  elif args.method == 'neural':
    from neuralNetworkClassifier import NeuralNetwork
    classifier = NeuralNetwork(data.X, data.Y, labels, args.avoid_skewness)
  elif args.method == 'baseline':
    classifier = Baseline(data.X_train, data.X_test, data.Y_train, data.Y_test, data.labels)

  classifier.classify()
  classifier.evaluate()
  classifier.printBasicEvaluation()

  data.writer('./test_00/', classifier.classifier)
  #classifier.printClassEvaluation()
  
else:
  print('The combination of the language <{}> and the variable <{}> only have one label. Thus, there is nothing to train. Try another combination!'.format(predict_languages, args.predict_label))