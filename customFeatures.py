import numpy as np
import re
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn import preprocessing

class CustomFeatures:
  class wordCount(TransformerMixin):
    def fit(self, X, y=None):
      return self

    def transform(self, X):
      newX = []
      for x in X:
        newX.append(len(x.split(' ')))
      return np.transpose(np.matrix(newX))

  class characterCount(TransformerMixin):
    def fit(self, X, y=None):
      return self

    def transform(self, X):
      newX = [len(x) for x in X]
      return np.transpose(np.matrix(newX))

  class userMentions(TransformerMixin):
    def fit(self, X, y=None):
      return self

    def transform(self, X):
      newX = []
      for x in X:
        user_counter = 0
        tokens = x.split(' ')
        for token in tokens:
          if '@username' in token:
            user_counter += 1
        newX.append(user_counter)
      return np.transpose(np.matrix(newX))

  class urlMentions(TransformerMixin):
    def fit(self, X, y=None):
      return self

    def transform(self, X):
      newX = []
      for x in X:
        url_counter = 0
        tokens = x.split(' ')
        for token in tokens:
          if 'url' in token:
            url_counter += 1
        newX.append(url_counter)
      return np.transpose(np.matrix(newX))

  class hashtagUse(TransformerMixin):
    def fit(self, X, y=None):
      return self

    def transform(self, X):
      newX = []
      for x in X:
        hashtag_counter = 0
        tokens = x.split(' ')
        for token in tokens:
          if '#' in token:
            hashtag_counter += 1
        newX.append(hashtag_counter)
      return np.transpose(np.matrix(newX))

  class sentiment(TransformerMixin):
    def fit(self, X, y=None):
      return self
      
    def transform(self, X):
      newX = []
      sid = SentimentIntensityAnalyzer()
      for x in X:
        newX.append(round(sid.polarity_scores(x)['pos']-sid.polarity_scores(x)['neg'], 2))
      return np.transpose(np.matrix(newX))

  class emoticonUse(TransformerMixin):
    def fit(self, X, y=None):
      return self

    def transform(self, X):
      newX = []
      for x in X:
          #print(token)
        emoticon_counter = len(re.findall(r'(:\(|:\))', x))
        if emoticon_counter > 0:
          emoticon_counter = 1
        newX.append(emoticon_counter)
      return np.transpose(np.matrix(newX))

      