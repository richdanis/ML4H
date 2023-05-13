import nltk
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
import pandas as pd
import string
import re
import math
import emoji
import contractions
from tqdm import tqdm

def remove_nans(data):

    return data[data['TweetText'].notnull()]

def remove_duplicates(data):

    # if there are duplicate of sentiment and tweet, then only keep one
    data = data.drop_duplicates(subset = ['TweetText', 'Sentiment'], keep = 'first')
    
    # if there are duplicate tweets left, that means that they have
    # different sentiments, in which case we remove them
    return data.drop_duplicates(subset = ['TweetText'], keep = False)

def remove_url(tweet):

    # remove hyperlinks
    # source: https://github.com/vasisouv/tweets-preprocessor/blob/master/twitter_preprocessor.py
    return re.sub(r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]'
           r'[a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:'
           r'\/\/(?:www\.|(?!www))[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9]\.[^\s]{2,})', '', tweet)

def replace_emojis(tweet):

    # replace emojis with their meaning
    return emoji.demojize(tweet, delimiters=(" ", " "))

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

    # replace underscore between emoji words
    tweet = tweet.replace("_", " ")

    # do not remove @, as it is handled by the tokenizer
    punctuation = string.punctuation.replace("@", "")

    # remove punctuation
    return tweet.translate(str.maketrans('', '', punctuation))

def remove_symbols(row):

    tweet = row['TweetText']
    tweet = remove_url(tweet)
    tweet = replace_emojis(tweet)
    tweet = remove_emojis(tweet)
    tweet = contractions.fix(tweet, slang=False)
    tweet = remove_punctuation(tweet)

    return tweet

def tokenization(tweets):

    tokenized_tweets = []
    # preserve case = False => convert to lowercase
    # strip_handles = True => remove @mentions
    # reduce_len = True => reduce length of repeated characters
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)

    for tweet in tqdm(tweets, desc="Tokenizing"):
        tokenized_tweets.append(tokenizer.tokenize(tweet))

    return tokenized_tweets

def lemmatization(tweets):

    # handle declination of words

    lemmatized_tweets = []

    lemmatizer = WordNetLemmatizer()

    for tweet in tqdm(tweets, desc="Lemmatizing"):
        lemmatized_tweets.append([lemmatizer.lemmatize(word) for word in tweet])

    return lemmatized_tweets

def preprocess(data, split=True):

    # download for lemmatization
    nltk.download('wordnet')

    # remove NaNs
    data = remove_nans(data)

    # progress bar
    tqdm.pandas(desc="Removing and Replacing Symbols")

    # remove some symbols
    data = data.assign(TweetText=data.progress_apply(remove_symbols, axis=1))

    # tokenize
    tweets = data['TweetText'].tolist()
    tokenized_tweets = tokenization(tweets)

    # lemmatize
    lemmatized_tweets = lemmatization(tokenized_tweets)

    joined = []
    for tweet in lemmatized_tweets:
        joined.append(' '.join(tweet))

    data = data.assign(TweetText=joined)

    # remove NaNs again
    data = remove_nans(data)

    # remove duplicates
    if split:
        data = remove_duplicates(data, split)

    # split sentiment
    if split:
        data = data.assign(PosSentiment=data['Sentiment'].apply(lambda x: x.split()[0]))
        data = data.assign(NegSentiment=data['Sentiment'].apply(lambda x: x.split()[1]))

    return data

def make_split(data):
  # makes 80/10/10 train/val/test split
  # and saves them in the data folder

  # shuffle the rows since ordered by date
  data = data.sample(frac=1, random_state=42)

  # get the classes
  classes = pd.unique(data['Sentiment']).tolist()

  test = val = train = None

  for c in classes:

    select = data[data['Sentiment'] == c]
    up = math.ceil(0.1 * select.shape[0])
    if test is None:
      test = select[:up]
      val = select[up:2*up]
      train = select[2*up:]
    else:
      test = pd.concat([test, select[:up]], ignore_index=True)
      val = pd.concat([val, select[up:2*up]], ignore_index=True)
      train = pd.concat([train, select[2*up:]], ignore_index=True)

  # shuffle 
  test = test.sample(frac=1, random_state=42)
  val = val.sample(frac=1, random_state=42)
  train = train.sample(frac=1, random_state=42)

  test.to_csv('data/test.csv', index=False)
  val.to_csv('data/val.csv', index=False)
  train.to_csv('data/train.csv', index=False)

def class_counts(data):
  # can be used to check the correctness of the split
  sentiments = data['Sentiment'].tolist()
  comb_counts = dict()
  for s in sentiments:
    if s in comb_counts:
      comb_counts[s] += 1
    else:
      comb_counts[s] = 1
  print(comb_counts)


if __name__ == '__main__':

    # read data
    data = pd.read_csv('data/TweetsCOV19.csv')

    # preprocess
    data = preprocess(data)

    # save to csv
    data.to_csv('data/cleaned_tweets.csv', index=False)
    
    # split into train/val/test and save in data folder
    make_split(data)