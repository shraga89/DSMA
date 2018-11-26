from keras.models import Sequential
from keras.layers import GRU, LSTM, Dense, TimeDistributed, Activation, Bidirectional, RepeatVector, Flatten, Permute, \
    Dropout, Lambda, Reshape, UpSampling1D
from keras.layers import Convolution2D as Conv2D
from keras.layers import Convolution1D as Conv1D
from keras.layers import GlobalMaxPooling2D
from keras.layers import GlobalMaxPooling1D, MaxPooling1D
from keras.layers import BatchNormalization
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_fscore_support
import tensorflow as tf
from theano.tensor.nnet.abstract_conv import border_mode_to_pad


def build_gru(first):
    model = Sequential()
    model.add(GRU(first, input_shape=(None, 1), return_sequences=True))
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))
    gru1 = GRU(first, input_shape=(None, 1), return_sequences=True, activation='relu')
    model.add(gru1)
    td1 = TimeDistributed(Dense(1, activation='sigmoid'))
    model.add(td1)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    print(model.summary())
    return model


# def build_cnn(first):
#     model = Sequential()
#     model.add(Conv1D(first, 16, input_shape=(None, 1), kernel_initializer='ones',
#                      activation='relu', padding="same"))  # single stride 4x4 filter for 16 maps
#     model.add(Conv1D(first * 2, 16, activation='sigmoid', padding="same"))  # single stride 4x4 filter for 32 maps
#     model.add(Dropout(0.5))
#     model.add(UpSampling1D())
#     # model.add(MaxPooling1D(padding="same", pool_size=2))
#     model.add(GlobalMaxPooling1D())
#     # model.add(BatchNormalization())
#     td1 = TimeDistributed(Dense(1, activation='sigmoid'))
#     model.add(td1)
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
#     print(model.summary())
#     return model

def build_cnn(first):
    model = Sequential()
    model.add(Conv1D(first, kernel_size=4, input_shape=(None, 1), padding="same", kernel_initializer='ones'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(UpSampling1D())
    model.add(Conv1D(first, kernel_size=4, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(Conv1D(1, kernel_size=4, padding="same"))
    model.add(MaxPooling1D(padding="same", pool_size=2))
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))
    # model.add(GlobalMaxPooling1D())
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    # print(model.summary())
    return model


def build_dnn(first):
    model = Sequential()
    model.add(Dense(first, input_shape=(None, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(first, activation='relu'))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    print(model.summary())
    return model


def build_cnn_gru(first):
    model = Sequential()
    model.add(Conv1D(first, kernel_size=4, input_shape=(None, 1), padding="same", kernel_initializer='ones'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(UpSampling1D())
    model.add(Conv1D(first, kernel_size=4, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(Conv1D(1, kernel_size=4, padding="same"))
    model.add(MaxPooling1D(padding="same", pool_size=2))
    model.add(GRU(first*2, return_sequences=True))
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    # print(model.summary())
    return model
