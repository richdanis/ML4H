import nltk
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
import pandas as pd
import string
import re

def remove_nans(data):

    return data[data['TweetText'].notnull()]

def remove_url(tweet):

    # remove hyperlinks
    # source: https://github.com/vasisouv/tweets-preprocessor/blob/master/twitter_preprocessor.py
    return re.sub(r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]'
           r'[a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:'
           r'\/\/(?:www\.|(?!www))[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9]\.[^\s]{2,})', '', tweet)

def remove_emojis(tweet):

    # remove emojis
    # source: https://stackoverflow.com/questions/33404752/removing-emojis-from-a-string-in-python
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoji_pattern, '', tweet)

def remove_punctuation(tweet):

    # remove punctuation
    return tweet.translate(str.maketrans('', '', string.punctuation))

def remove_symbols(row):

    tweet = row['TweetText']
    tweet = remove_url(tweet)
    tweet = remove_emojis(tweet)
    tweet = remove_punctuation(tweet)

    return tweet

def tokenization(tweets):

    tokenized_tweets = []
    # preserve case = False => convert to lowercase
    # strip_handles = True => remove @mentions
    # reduce_len = True => reduce length of repeated characters
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)

    for tweet in tweets:
        tokenized_tweets.append(tokenizer.tokenize(tweet))

    return tokenized_tweets

def lemmatization(tweets):

    # handle declination of words

    lemmatized_tweets = []

    lemmatizer = WordNetLemmatizer()

    for tweet in tweets:
        lemmatized_tweets.append([lemmatizer.lemmatize(word) for word in tweet])

    return lemmatized_tweets

def preprocess(data):

    # download for lemmatization
    nltk.download('wordnet')

    # remove NaNs
    data = remove_nans(data)

    # remove some symbols
    data['TweetText'] = data.apply(remove_symbols, axis=1)

    # tokenize
    tweets = data['TweetText'].tolist()
    tokenized_tweets = tokenization(tweets)

    # lemmatize
    lemmatized_tweets = lemmatization(tokenized_tweets)

    joined = []
    for tweet in lemmatized_tweets:
        joined.append(' '.join(tweet))

    data['TweetText'] = joined

    return data