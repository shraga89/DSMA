from keras.models import Sequential
from keras.layers import GRU, LSTM, Dense, TimeDistributed, Activation, Bidirectional, RepeatVector, Flatten, Permute, \
    Dropout, Lambda, Reshape, Embedding
from keras.layers import Convolution2D as Conv2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import Convolution1D as Conv1D
from keras.layers import GlobalMaxPooling1D, MaxPooling1D
from keras.layers import BatchNormalization
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_fscore_support
import tensorflow as tf

loss = 'mean_squared_error'
optimizer = 'adam'


def build_gru(first):
    model = Sequential()
    model.add(GRU(first, input_shape=(None, 1), return_sequences=True))
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))
    model.add(GRU(first, input_shape=(None, 1), return_sequences=False, activation='softmax'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss=loss, optimizer=optimizer, metrics=["acc"])
    print(model.summary())
    return model


def build_cnn_2d(first):
    model = Sequential()
    model.add(Conv2D(first, (4, 4), input_shape=(None, None, 1), activation='relu'))  # single stride 4x4 filter for 16 maps
    model.add(Conv2D(first * 2, (4, 4), activation='relu'))  # single stride 4x4 filter for 32 maps
    model.add(Dropout(0.5))
    model.add(GlobalMaxPooling2D())
    model.add(BatchNormalization())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss=loss, optimizer=optimizer, metrics=["acc"])
    print(model.summary())
    return model


def build_cnn(first):
    model = Sequential()
    model.add(Conv1D(first, 16, input_shape=(None, 1), activation='relu'))  # single stride 4x4 filter for 16 maps
    model.add(Conv1D(first * 2, 16, activation='relu'))  # single stride 4x4 filter for 32 maps
    model.add(Dropout(0.5))
    model.add(GlobalMaxPooling1D())
    model.add(BatchNormalization())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss=loss, optimizer=optimizer, metrics=["acc"])
    print(model.summary())
    return model


def build_dnn(first):
    model = Sequential()
    model.add(Dense(first, input_shape=(None, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(first, activation='relu'))
    model.add(Dropout(0.5))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss=loss, optimizer=optimizer, metrics=['acc'])
    print(model.summary())
    return model


def build_cnn_gru(first):
    model = Sequential()
    model.add(Conv1D(first, 16, input_shape=(None, 1), activation='relu'))
    model.add(MaxPooling1D(16))
    model.add(GRU(first*2, return_sequences=False))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss=loss, optimizer=optimizer, metrics=['acc'])
    # print(model.summary())
    return model
