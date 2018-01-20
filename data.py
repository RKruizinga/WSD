import os
import xml.etree.ElementTree as ET

import re
class data:

  X_train = []
  Y_train = []
  words_train = []
  labels_train = []

  X_test = []
  Y_test = []
  words_test = []
  labels_test = []

  docs_train = []
  docs_test = []


  def __init__(self, train_dir, test_dir):
    self.train_dir = train_dir
    self.test_dir = test_dir

    self.readFiles()

  def readFiles(self):
    train_dir = os.listdir(self.train_dir)

    for cur_file in train_dir:
      with open(self.train_dir + cur_file, encoding="utf8") as f:
        document = []
        data = f.read()
        data = data.split('\n')
        
        for row in data:
          row = row.split('\t')
          row[0] = row[0].split(' ')
          if len(row[0]) > 1:
            document.append(row)
        self.docs_train.append(document)
    
    test_dir = os.listdir(self.test_dir)
    for cur_file in test_dir:
      with open(self.test_dir + cur_file, encoding="utf8") as f:
        document = []
        data = f.read()
        data = data.split('\n')
        
        for row in data:
          row = row.split('\t')
          row[0] = row[0].split(' ')
          if len(row[0]) > 1:
            document.append(row)
        self.docs_test.append(document)

  def collector(self, documents):

    X = []
    Y = []
    words = []
    labels = []
    
    for document in documents:
      text = []
      for token in document: #split text for particular sentences, to get the right context for ambiguous words
        if re.match('(.)?(.001)$', token[0][2]):
          text.append([])
          text[len(text)-1].append(token[0][3])
        else:
          text[len(text)-1].append(token[0][3])
      #text = [token[0][3] for token in document]
      for token in document:
        if len(token) == 2: 
          if int(token[0][2]) > 10000:
            X.append(' '.join(text[int(token[0][2][0:2])-1]))
          else:
            X.append(' '.join(text[int(token[0][2][0])-1]))#select the right sentence for the word  
          
          word = token[0][3].lower()
          word = re.sub(r'(\w+)es$', r'\1', word)
          word = re.sub(r'(\w+)s$', r'\1', word)

          label = token[1]
          words.append(word)
          labels.append(label)
          Y.append(word+'_'+label)
    return X,  Y, words, labels

  def collectXY(self, data_method = 1):
    self.X_train, self.Y_train, self.words_train, self.labels_train = self.collector(self.docs_train)
    self.X_test, self.Y_test, self.words_test, self.labels_test = self.collector(self.docs_test)

    self.X = self.X_train + self.X_test
    self.Y = self.Y_train + self.Y_test

    self.words = self.words_train + self.words_test
    self.labels = self.labels_train + self.labels_test

  def writer(self, documents, classifier):
    test_dir = os.listdir(self.test_dir)
    for cur_file in test_dir:

      with open(self.test_dir + cur_file, encoding="utf8") as f:
        new_document = []
        document = []
        data = f.read()
        data = data.split('\n')
        
        for row in data:
          row = row.split('\t')
          row[0] = row[0].split(' ')

          if len(row[0]) > 1:
            document.append(row)
        text = []

        for token in document: #split text for particular sentences, to get the right context for ambiguous words
          if re.match('(.)?(.001)$', token[0][2]):
            text.append([])
            text[len(text)-1].append(token[0][3])
          else:
            text[len(text)-1].append(token[0][3])
        #text = [token[0][3] for token in document]
        for token in document:
          if len(token) == 2: 
            if int(token[0][2]) > 10000:
              X = ' '.join(text[int(token[0][2][0:2])-1])
            else:
              X = ' '.join(text[int(token[0][2][0])-1])
            word = token[0][3].lower()
            word = re.sub(r'(\w+)s$', r'\1', word)
            word = re.sub(r'(\w+)s$', r'\1', word)

            label = token[1]

            Y_predicted = classifier.predict([X])
            Y_predicted_sense = Y_predicted[0].split('_')[1]
            new_document.append(' '.join(token[0])+'\t'+str(Y_predicted_sense))
          else:
            new_document.append(' '.join(token[0]))

        new_document = '\n'.join(new_document)
        file = open('./test_00_annotated/'+ cur_file, 'w')
        file.write(new_document) 
        file.close() 



