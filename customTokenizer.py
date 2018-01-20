from nltk.tokenize import TweetTokenizer
from nltk import word_tokenize

class CustomTokenizer: #collection class of different tokenizers
  def tweetIdentity(arg):
	  tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)
	  return tokenizer.tokenize(arg)

  def wordTokenize(arg):
    return word_tokenize(arg)
    
  def tokenizeTweets(tweets):
    tokenized_tweets = []
    #since the Tokenizer of keras expects a text, we tokenize the tweets, but also join it together as a string
    tweetTokenizer = TweetTokenizer()

    for tweet in tweets:
      tokenized_tweet = '|'.join(tweetTokenizer.tokenize(tweet))
      tokenized_tweets.append(tokenized_tweet)
      
    return tokenized_tweets
