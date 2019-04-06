import numpy as np
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

dataframe = pd.read_csv("Processed_Tweets/clean_tweets.csv", index_col=0)
negative_tweets = dataframe[dataframe.target == 0]
negative_tweets_list = []
for tweet in negative_tweets.text:
    negative_tweets_list.append(tweet)
negative_tweets_list = pd.Series(negative_tweets_list).str.cat(sep=' ')

wordcloud = WordCloud(width=1600, height=800, max_font_size=200, colormap='viridis').generate(negative_tweets_list)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.savefig("Visualization/Negative_Tweets.png")
positive_tweets = dataframe[dataframe.target == 1]
positive_tweets_list = []
for tweet in positive_tweets.text:
    positive_tweets_list.append(tweet)
positive_tweets_list = pd.Series(positive_tweets_list).str.cat(sep=' ')

wordcloud = WordCloud(width=1600, height=800, max_font_size=200, colormap='inferno').generate(positive_tweets_list)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig("Visualization/Positive_Tweets.png")
plt.show()
