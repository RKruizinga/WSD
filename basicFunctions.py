from collections import Counter
import random
import numpy as np

import sklearn
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

class BasicFunctions:
  def avg(l):
    return sum(l) / len(l)

  def keyCounter(list):
    return Counter(list)

  def printStandardText(method, languages, variable):
    print('#'*91)

    if variable == 'age':
      printVar = '\t\t\t\t\t\t'
    elif len(variable) > 20:
      printVar = '\t\t\t'
    else:
      printVar = '\t\t\t\t\t'
    
    if len(languages) == 4:
      printLan = '\t\t'
    elif len(languages) == 3:
      printLan = '\t\t\t'
    elif len(languages) == 2:
      printLan = '\t\t\t\t'
    elif len(languages) == 1:
      printLan = '\t\t\t\t\t'

    if method == 'neural' or method == 'baseline':
      printMethod = '\t\t\t\t'
    else:
      printMethod = '\t\t\t\t\t'

    print('{} \t LFD Classification Output \t\t\t\t\t {}'.format('#'*10, '#'*10))
    print('{} \t Machine Learning Method: {} {} {}'.format('#'*10, method, printMethod, '#'*10))
    print('{} \t Predict Language(s): {} {} {}'.format('#'*10, ', '.join(languages), printLan, '#'*10))
    print('{} \t Predict Variable: {} {} {}'.format('#'*10, variable, printVar, '#'*10))
    print('#'*91)

  def printEvaluation(accuracy, precision, recall, f1score, text):    
    print("~~~" + text + "~~~ \n")
    print("Accuracy:\t {}".format(round(accuracy, 3)))
    print("Precision:\t {}".format(round(precision, 3)))
    print("Recall:\t\t {}".format(round(recall, 3)))
    print("F1-Score:\t {}".format(round(f1score, 3)))
  
  def printClassEvaluation(Y_test, Y_predicted, labels):

    print("\n~~~ Class Evaluation ~~~ \n")
    print("Class \t Precision \t Recall \t F-score")

    for label in labels:
      accuracy, precision, recall, f1score = BasicFunctions.getMetrics(Y_test, Y_predicted, [label])
      print('{} \t {} \t\t {} \t\t {}'.format(
        label,
        round(precision, 3),
        round(recall, 3),
        round(f1score, 3)
      ))
  def printLabelDistribution(labels):
    label_distribution = BasicFunctions.keyCounter(labels)

    print('~~~Label Distribution~~~')
    for label in label_distribution:
      print('{} \t {}'.format(label, label_distribution[label]))

  def getMetrics(Y_test, Y_predicted, labels):
    accuracy_count = 0
    for i in range(0, len(Y_predicted)):
      if Y_predicted[i] == Y_test[i]:
        accuracy_count += 1
    accuracy = accuracy_count/len(Y_predicted)

    already_set = False
    clean_labels = [] #without errors
    if len(labels) == 1:
      if labels[0] not in Y_predicted:
        precision = 0.0
        recall = 0.0
        f1score = 0.0
        already_set = True
      clean_labels.append(labels[0])
    else:
      for label in labels:
        if label in Y_predicted:
          clean_labels.append(label)

    if already_set == False:
      precision = sklearn.metrics.precision_score(Y_test, Y_predicted, average="macro", labels=clean_labels)
      recall = sklearn.metrics.recall_score(Y_test, Y_predicted, average="macro", labels=clean_labels)
      f1score = sklearn.metrics.f1_score(Y_test, Y_predicted, average="macro", labels=clean_labels)

    return accuracy, precision, recall, f1score

  def getLanguages(argument_languages):
    possible_languages = ['dutch', 'english', 'spanish', 'italian']
    if argument_languages == 'all':
      predict_languages = possible_languages

    predict_languages = argument_languages.split(',')

    new_format = []
    if len(predict_languages) == 1:
      if predict_languages[0] not in possible_languages:
        for letter in predict_languages[0]:
          for possible_language in possible_languages:
            if letter == possible_language[0]:
              new_format.append(possible_language)
        predict_languages = new_format
    return predict_languages

  def getUnskewedSubset(X_train, Y_train, Y_train_raw = None):
    if Y_train_raw == None:
      data_distribution = BasicFunctions.keyCounter(Y_train)
      Y = Y_train
    else:
      data_distribution = BasicFunctions.keyCounter(Y_train_raw)
      Y = Y_train_raw

    lowest_amount = 0
    for label in data_distribution:
      if data_distribution[label] < lowest_amount or lowest_amount == 0:
        lowest_amount = data_distribution[label]
    key_dict = {}
    for i, label in enumerate(Y):
      if label not in key_dict:
        key_dict[label] = [i]
      else:
        key_dict[label].append(i)

    new_X_train = []
    new_Y_train = []
    all_keys = []
    new_dict = {}
    for label in key_dict: 
      new_dict[label] = random.sample(key_dict[label], lowest_amount)
      all_keys += new_dict[label]
    for i in sorted(all_keys):
      new_X_train.append(X_train[i])
      new_Y_train.append(Y_train[i])

    return new_X_train, new_Y_train