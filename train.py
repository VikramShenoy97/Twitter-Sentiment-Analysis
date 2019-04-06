import numpy as np
from load_data import load_dataset
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.embeddings import Embedding

train_X, train_y = load_dataset(mode="train")
embedding_vector_length = 128
input_len = len(train_X[0])
vocabulary_size = 271310
model = Sequential()
model.add(Embedding(vocabulary_size, embedding_vector_length, input_length=input_len))
model.add(LSTM(units=100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
print model.summary()
model.fit(train_X, train_y, batch_size=64, epochs=1)
model.save("Saved_Files/Model.h5")
