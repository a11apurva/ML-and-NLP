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

_corpus='train-hashed.txt'
_sep='\t'

df = pd.read_csv(_corpus, sep = _sep, names = ['labels', 'text'], error_bad_lines=False, encoding = "ISO-8859-1")
df= df.dropna()
df = df.sample(frac=1).reset_index(drop=True)

print("\n")
print(df.describe())
print("\n")
print(df.head(10))
print("\n")


###
t = Tokenizer(lower=False)
t.fit_on_texts(df['text'])
vocab_size = len(t.word_index) + 1

encoded_docs = t.texts_to_sequences(df['text'])


###
max_length = 10
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

print(padded_docs.shape)



###
lbl=set()
for line in open(_corpus,"r"):
    line=line.strip().split()
    lbl.add(line[0])

classes=len(lbl)

labels = df['labels']
one_hot_labels = utils.to_categorical(labels, num_classes=classes)


###
embeddings_index = dict()
f = open('model_component_hashed.vec')
for line in f:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()

print('Loaded %s word vectors.' % len(embeddings_index))
# create a weight matrix for words in training docs
embedding_matrix = np.zeros((vocab_size, 100))
for word, i in t.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector
   
   
###Training
model = Sequential()
e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=10, trainable=False)
model.add(e)
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(classes, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

print(model.summary())
history = model.fit(padded_docs, one_hot_labels, epochs=16, verbose=1 , validation_split=0.2)

###
loss, accuracy = model.evaluate(padded_docs, one_hot_labels, verbose=1)
print('Accuracy: %f' % (accuracy*100))
 

###
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
   
###Validation
_corpus="validate-hashed.txt"
_sep="\t"
df2 = pd.read_csv(_corpus, sep = _sep, names = ['labels', 'text'], error_bad_lines=False)

encoded_docs_validate = t.texts_to_sequences(df2['text'])
max_length = 10
x_test = pad_sequences(encoded_docs_validate, maxlen=max_length, padding='post')

labels = df2['labels']
y_test = utils.to_categorical(labels, num_classes=classes)

score = model.evaluate(x_test, y_test)

print('Test score:', score[0])
print('Test accuracy:', score[1])
   
   
   
   
   
