import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def filter_tweets(tweets, lowercase=True):
    mod_tweets = list()

    for tweet in tweets:
        mod_tweet = tweet.lower() if lowercase else tweet
        mod_tweet = mod_tweet.replace('#url#', '')
        mod_tweet = mod_tweet.replace('#user#', '')
        mod_tweet = mod_tweet.replace('#hashtag#', '')
        mod_tweet = mod_tweet.replace('rt', '')

        mod_tweets.append(mod_tweet)

    return mod_tweets


def group_tweets_by_author(tweets, usr_len=200):
    n_authors = int(len(tweets) / usr_len)
    grouped_tweets = list()

    for i in range(n_authors):
        start = i * usr_len
        author_tweets = ' '.join(tweets[start: start + usr_len])
        grouped_tweets.append(author_tweets)

    return grouped_tweets


def prepare_authlist(authors, usr_len=200):
    n_vals = int(len(authors) / usr_len)
    authlist = list()

    for i in range(n_vals):
        authlist.append(authors[i * usr_len].strip())

    return authlist


def prepare_labels(labels, usr_len=200):
    n_vals = int(len(labels) / usr_len)
    y_labels = np.zeros(n_vals)

    for i in range(n_vals):
        y_labels[i] = labels[i * usr_len]

    return y_labels


def prepare_tweets_using_tfidf(tweets, lang, is_train=True, vec=None):
    if is_train:
        vec = TfidfVectorizer(stop_words='english') if lang == 'en' else TfidfVectorizer()
        tweet_features = vec.fit_transform(tweets)
    else:
        tweet_features = vec.transform(tweets)

    return tweet_features, vec


def prepare_xy(df, tweet_feature_method, lang, return_y=True, usr_len=200, is_train=True, vec=None):
    tweets = df['tweet'].to_list()
    tweets = filter_tweets(tweets)
    auth_tweets = group_tweets_by_author(tweets, usr_len)
    x, vec = tweet_feature_method(auth_tweets, lang, is_train, vec)

    if return_y:
        labels = df['label'].to_list()
        y = prepare_labels(labels, usr_len)
        return x, y, vec
    else:
        return x, vec
