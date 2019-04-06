# Twitter Sentiment Analysis

Twitter Sentiment Analysis using Recurrent Neural Networks with LSTM units on Sentiment140 Dataset.


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

For using this project on sentiment analysis, you need to install Keras, Scikit-learn, The Natural Language Toolkit (NLTK), BeautifulSoup, and WordCloud.

```
pip install keras
pip install scikit-learn
pip install nltk
pip install bs4
pip install wordcloud
```

### Dataset
The dataset consists of 1.6 million unprocessed tweets from the [Sentiment140 Dataset](http://help.sentiment140.com/for-students) with labels 0 = Negative and 4 = Positive.

The dataset is stored in the folder **Unprocessed_Data** as *training.1600000.processed.noemoticon.csv* *(File too large to upload onto Github)*.

### Data Preprocessing

Run the script *tweet_cleaner.py* in the terminal as follows.

```
Python tweet_cleaner.py
```

• This script saves the tweets as *clean_tweets.csv* in the **Processed_Tweets** folder *(File too large to upload onto Github)*.

• Splits the data into Training Set (98%) and Testing Set (2%). The data and labels are stored in the **Saved_Files** folder. *(train_data.npy too large to upload onto Github)*.

• It also creates the vocabulary (Dictionary format) and stores it in *vocabulary.npy* in the **Saved_Files** folder.

### Word Cloud Visualization (Optional)

To view the most common words in positve and negative tweets, run the script *word_cloud_visualization.py* as follows.

```
Python word_cloud_visualization.py
```

It outputs the following two images (Stored in the **Visualization** folder):

##### Words from Positive Tweets

![positive](https://github.com/VikramShenoy97/Twitter-Sentiment-Analysis/blob/master/Visualization/Positive_Tweets.png)


##### Words from Negative Tweets

![positive](https://github.com/VikramShenoy97/Twitter-Sentiment-Analysis/blob/master/Visualization/Negative_Tweets.png)

### Training

98% of the data is trained using a Recurrent Neural Network with 100 Long Short-Term Memory Units. 
#### Neural Network Model

![model](https://github.com/VikramShenoy97/Twitter-Sentiment-Analysis/blob/master/Saved_FIles/model.png)

Run the script *train.py* in the terminal as follows.

```
Python train.py
```

The model with trained weights is saved as *Model.h5* in the **Saved_Files** folder. *(Model.h5 has over 34 million parameters and hence is too large to upload onto Github)*.

### Testing

Run the script *test.py* in the terminal as follows.

```
Python test.py
```

After evaluating the test accuracy, the code gives you an option to try out Sentiment Analysis on custom text (The words of the custom text should be a part of the vocabulary).

## Results

### Final Accuracy

```
Loss on Test Set: 0.3491
Accuracy on Test Set: 84.5756%
```

#### Prediction Graph

Visualization of the predictions is stored in the **Output** folder. 10% of the predictions (Approximately 3000 points) are shown in the graph.

![graph](https://github.com/VikramShenoy97/Twitter-Sentiment-Analysis/blob/master/Output/Predictions.png)

#### Sentiment Analysis

An example of custom text sentiment analysis.

![test](https://github.com/VikramShenoy97/Twitter-Sentiment-Analysis/blob/master/Output/test.png)


## Built With

* [Keras](https://keras.io) - Deep Learning Framework

## Authors

* **Vikram Shenoy** - *Initial work* - [Vikram Shenoy](https://github.com/VikramShenoy97)

## Acknowledgments

* Project is inspired by **Ricky Kim's** blog, [*Another Twitter Sentiment Analysis with Python.*](https://towardsdatascience.com/another-twitter-sentiment-analysis-bb5b01ebad90)
