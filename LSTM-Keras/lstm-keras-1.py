from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras import utils
import plotly.offline as py
import plotly.graph_objs as go
import nltk
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

_corpus='train.txt'
_sep='\t'

#_corpus='data1.txt'
#_sep=','

df = pd.read_csv(_corpus, sep = _sep, names = ['labels', 'text'], error_bad_lines=False, encoding = "ISO-8859-1")
df= df.dropna()
df = df.sample(frac=1).reset_index(drop=True)

print("\n")
print(df.describe())
print("\n")
print(df.head(10))
print("\n")

vocabulary_size = 200000
tokenizer = Tokenizer(num_words= vocabulary_size)
tokenizer.fit_on_texts(df['text'])

sequences = tokenizer.texts_to_sequences(df['text'])

sum=0

for seq in sequences:
    sum += len (seq)
    


data = pad_sequences(sequences, maxlen= avg)

print(data.shape)

lbl=set()
for line in open(_corpus,"r"):
    line=line.strip().split()
    lbl.add(line[0])

classes=len(lbl)
 
labels = df['labels']
one_hot_labels = utils.to_categorical(labels, num_classes=classes)

model_lstm = Sequential()
model_lstm.add(Embedding(vocabulary_size, 300, input_length=avg))
model_lstm.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model_lstm.add(Dense(classes, activation='softmax'))
model_lstm.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model_lstm.fit(data, one_hot_labels, epochs=8, validation_split=0.0)

print("\n***History***")
print(history.history) 

print("\n***Evaluating***")

#vocabulary_size = 2000000
#tokenizer = Tokenizer(num_words= vocabulary_size)
#test_data=['vv component failed termerror code 0x1a',  'the customer found a network port failure']
#tokenizer.fit_on_texts(df['text'])
#sequences = tokenizer.texts_to_sequences(test_data)
#data = pad_sequences(sequences, maxlen= avg)

#prediction = model_lstm.predict(data)
#print(prediction)

#ynew = model_lstm.predict_classes(data)
#for i in range(len(data)):
#    print("X=%s, Predicted=%s" % (test_data[i], ynew[i]))

_corpus="validate.txt"
_sep="\t"
df2 = pd.read_csv(_corpus, sep = _sep, names = ['labels', 'text'], error_bad_lines=False)

#vocabulary_size = 200000
#tokenizer = Tokenizer(num_words= vocabulary_size)
#tokenizer.fit_on_texts(df['text'])

sequences = tokenizer.texts_to_sequences(df2['text'])
x_test = pad_sequences(sequences, maxlen= avg)

labels = df2['labels']
y_test = utils.to_categorical(labels, num_classes=classes)

score = model_lstm.evaluate(x_test, y_test)

print('Test score:', score[0])
print('Test accuracy:', score[1])

plt.subplot(211)
plt.title("accuracy")
plt.plot(history.history["acc"], color="r", label="train")
plt.plot(history.history["val_acc"], color="b", label="val")
plt.legend(loc="best")

plt.subplot(212)
plt.title("loss")
plt.plot(history.history["loss"], color="r", label="train")
plt.plot(history.history["val_loss"], color="b", label="val")
plt.legend(loc="best")

plt.tight_layout()
plt.show()

text1= ["enter text 1","enter text 2","and so on"]

#vocabulary_size = 200000
#tokenizer = Tokenizer(num_words= vocabulary_size)
#tokenizer.fit_on_texts(df['text'])
sequences = tokenizer.texts_to_sequences(text1)
X = pad_sequences(sequences, maxlen= avg)

yhat = model_lstm.predict(X, verbose=1)
print(yhat)
yhat = model_lstm.predict_classes(X)
print(yhat)
yhat = model_lstm.predict_proba(X)
print(yhat)

y_prob = model_lstm.predict(X) 
y_classes = y_prob.argmax(axis=-1)
