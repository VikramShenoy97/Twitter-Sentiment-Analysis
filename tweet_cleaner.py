import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import WordPunctTokenizer
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup

def cleaning_process(unprocessed_tweets_file):
    cols = ['sentiment','id','date','query_string','user','text']
    dataframe = pd.read_csv(unprocessed_tweets_file, header=None, names=cols)
    dataframe.drop(['id','date','query_string','user'], axis=1, inplace=True)
    dataframe['sentiment'] = dataframe['sentiment'].map({0: 0, 4: 1})
    processed_tweets = []
    for i in range(0, len(dataframe['text'])):
        processed_tweets.append(tweet_clean(dataframe['text'][i]))

    clean_tweets_dataframe = pd.DataFrame(processed_tweets, columns=['text'])
    clean_tweets_dataframe['target'] = dataframe.sentiment
    clean_tweets_dataframe.to_csv('Processed_Tweets/clean_tweets.csv', encoding='utf-8')
    return
    
def tweet_clean(tweet):
    condition_1 = r'@[A-Za-z0-9]+'
    condition_2 = 'https?://[A-Za-z0-9./]+'
    condition_3 = r'www.[^ ]+'
    negations_dictionary = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",
                    "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
                    "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
                    "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
                    "mustn't":"must not"}
    negations_pattern = re.compile(r'\b(' + '|'.join(negations_dictionary.keys()) + r')\b')
    tokenizer = WordPunctTokenizer()
    combined_conditions = r'|'.join((condition_1, condition_2, condition_3))
    processed_tweet = BeautifulSoup(tweet, "lxml")
    processed_tweet = processed_tweet.get_text()
    processed_tweet = re.sub(combined_conditions, '', processed_tweet)
    try:
        cleaned_tweet = processed_tweet.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        cleaned_tweet = processed_tweet

    cleaned_tweet = cleaned_tweet.lower()
    negation_handled = negations_pattern.sub(lambda x: negations_dictionary[x.group()], cleaned_tweet)
    cleaned_tweet = re.sub("[^A-Za-z]", ' ', negation_handled)
    words = [x for x in tokenizer.tokenize(cleaned_tweet) if len(x)>1]
    return (" ".join(words)).strip()

def create_dataset():
    dataframe = pd.read_csv("Processed_Tweets/clean_tweets.csv", index_col=0)
    dataframe.dropna(inplace=True)
    dataframe.reset_index(drop=True, inplace=True)
    dataframe.info()
    count_vector = CountVectorizer()
    count_vector.fit(dataframe.text)
    negative_doc_matrix = count_vector.transform(dataframe[dataframe.target == 0].text)
    positive_doc_matrix = count_vector.transform(dataframe[dataframe.target == 1].text)
    negative_sum = np.sum(negative_doc_matrix, axis=0)
    positive_sum = np.sum(positive_doc_matrix, axis=0)
    negative = np.squeeze(np.asarray(negative_sum))
    positive = np.squeeze(np.asarray(positive_sum))
    term_frequency_dataframe = pd.DataFrame([negative, positive], columns=count_vector.get_feature_names()).transpose()
    term_frequency_dataframe.columns = ["negative", "positive"]
    term_frequency_dataframe["total"] = term_frequency_dataframe["positive"] + term_frequency_dataframe["negative"]
    number_of_words = len(term_frequency_dataframe["total"])
    word_list = term_frequency_dataframe.sort_values(by='total', ascending=False).iloc[:number_of_words].axes[0].tolist()
    vocabulary = {key: value for value, key in enumerate(word_list, start=1)}
    tokenizer = WordPunctTokenizer()
    list_tweets = []
    for i in range(0, len(dataframe.text)):
        list_words = [word for word in tokenizer.tokenize(dataframe.text[i])]
        list_tweets.append(list_words)

    X_list = []
    y_list = []
    for i in range(0, len(list_tweets)):
        list_index = []
        for j in range(0, len(list_tweets[i])):
            list_index.append(vocabulary[list_tweets[i][j]])
        X_list.append(list_index)
        y_list.append(dataframe.target[i])

    X = np.array([np.array(x) for x in X_list])
    y = np.array(y_list)
    return X, y, vocabulary

unprocessed_tweets_file = "Unprocessed_Data/training.1600000.processed.noemoticon.csv"
cleaning_process(unprocessed_tweets_file)
X,y,vocabulary = create_dataset()

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.02, shuffle=True)

np.save("Saved_Files/train_data.npy", train_X)
np.save("Saved_Files/train_labels.npy", train_y)
np.save("Saved_Files/test_data.npy", test_X)
np.save("Saved_Files/test_labels.npy", test_y)
np.save("Saved_Files/vocabulary.npy", vocabulary)
