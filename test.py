import numpy as np
import keras
import re
from keras.models import Model, load_model
from keras.preprocessing import sequence
from load_data import load_dataset
from nltk import WordPunctTokenizer

from plotly import tools
import plotly.plotly as py
import plotly.graph_objs as go

def text_cleaner(text):
    negations_dictionary = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",
                    "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
                    "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
                    "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
                    "mustn't":"must not"}
    negations_pattern = re.compile(r'\b(' + '|'.join(negations_dictionary.keys()) + r')\b')
    tokenizer = WordPunctTokenizer()
    processed_text = text.lower()
    negation_handled = negations_pattern.sub(lambda x: negations_dictionary[x.group()], processed_text)
    processed_text = re.sub("[^A-Za-z]", ' ', negation_handled)
    words = [x for x in tokenizer.tokenize(processed_text) if len(x)>1]
    return words

def draw_graph(predictions, test_y):
    py.sign_in('VikramShenoy','x1Un4yD3HDRT838vRkFA')
    scatter_array_true = []
    scatter_array_false = []
    for i in range(len(predictions)):
        if round(predictions[i]) == test_y[i]:
            if(predictions[i] > 0.5):
                scatter_array_true.append(float(predictions[i])-0.5)
            else:
                scatter_array_true.append(float(predictions[i])-0.5)
        else:
            if(predictions[i] > 0.5):
                scatter_array_false.append(float(predictions[i])-0.5)
            else:
                scatter_array_false.append(float(predictions[i])-0.5)

    n = int(len(scatter_array_true)/10)
    random_x = np.random.randn(n)
    trace1 = go.Scatter(
    x = random_x,
    y = np.random.choice(scatter_array_true, n, replace=False),
    mode = 'markers',
    name = "Correct Prediction",
    marker = dict(
        size = 3,
        color = 'Green'))

    n = int(len(scatter_array_false)/10)
    random_x = np.random.randn(n)
    trace2 = go.Scatter(
    x = random_x,
    y = np.random.choice(scatter_array_false, n, replace=False),
    name = "Wrong Prediction",
    mode = 'markers',
    marker = dict(
        size = 3,
        color = 'Red'))
    data = [trace1, trace2]
    layout = dict(title = 'Prediction of Tweets')

    fig = dict(data=data, layout=layout)
    py.image.save_as(fig, filename="Output/Predictions.png")
    return

test_X, test_y = load_dataset(mode="test")

model = load_model("Saved_Files/Model.h5")
model.set_weights(model.get_weights())

print model.summary()

score = model.evaluate(test_X, test_y, verbose=1)
predictions=model.predict(test_X, verbose=0)
draw_graph(predictions, test_y)
print score

vocabulary = np.load("Saved_Files/vocabulary.npy").item()
continue_loop = "yes"
while(continue_loop == "yes"):
    test_string = raw_input("Enter Text for Analysis:\n")
    list_words = text_cleaner(test_string)
    list_index = []
    X_list = []
    for i in range(0, len(list_words)):
        list_index.append(vocabulary[list_words[i]])
    X_list.append(list_index)
    test = np.array(X_list)
    max_tweet_length = 40
    test = sequence.pad_sequences(test, maxlen=max_tweet_length)
    prediction = model.predict(test, verbose=0)
    predict = np.squeeze(prediction)
    if(predict > 0.5):
        print "Positive Sentiment with value = %0.2f" %(predict)
    else:
        print "Negative Sentiment with value = %0.2f" %(predict)
    continue_loop = raw_input("Do you want to continue? (yes / no):\n")
