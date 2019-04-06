import numpy as np
import keras
from keras.preprocessing import sequence

def load_dataset(mode=None):
    train_X = np.load("Saved_Files/train_data.npy")
    train_y = np.load("Saved_Files/train_labels.npy")
    test_X = np.load("Saved_Files/test_data.npy")
    test_y = np.load("Saved_Files/test_labels.npy")
    max_tweet_length = 40
    train_X = sequence.pad_sequences(train_X, maxlen=max_tweet_length)
    test_X = sequence.pad_sequences(test_X, maxlen=max_tweet_length)
    if(mode == "train"):
        return train_X, train_y
    elif(mode == "test"):
        return test_X, test_y
    else:
        return train_X, train_y, test_X, test_y
