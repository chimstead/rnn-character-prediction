# Keras character prediction model, used to learn and then generate sequences
# of characters that mimic the style, formatting, and logic of the input corpus

from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from keras.callbacks import ModelCheckpoint
import numpy as np
import random
import sys
import io

# read and prep the text and dictionaries
path = 'cards2.txt'
text = open(path).read()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_to_index = dict((c, i) for i, c in enumerate(chars))
index_to_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of sequence_length characters
sequence_length = 200
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - sequence_length, step):
    sentences.append(text[i: i + sequence_length])
    next_chars.append(text[i + sequence_length])
print('nb sequences:', len(sentences))

print('Vectorization...')
x = np.zeros((len(sentences), sequence_length, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_to_index[char]] = 1
    y[i, char_to_index[next_chars[i]]] = 1


# build the model
print('Build model...')
model = Sequential()
model.add(LSTM(256, return_sequences = True, input_shape=(sequence_length, len(chars))))
model.add(LSTM(153))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(learning_rate=0.005)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

# helper function to sample an index from a probability array
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# Function invoked at end of each epoch. Prints generated text.
def on_epoch_end(epoch, _):
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(text) - sequence_length - 1)
    for diversity in [0.1, 0.2, 0.3, 0.4, 0.5]:
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + sequence_length]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(700):
            x_pred = np.zeros((1, sequence_length, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_to_index[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = index_to_char[next_index]

            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()


# define the actions at end of each epoch
filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
callbacks_list = [checkpoint, print_callback]

# fit the model
model.fit(x, y,
          batch_size=128,
          epochs=15,
          callbacks=callbacks_list)
