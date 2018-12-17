from keras.models import Sequential, Model
from keras.layers import GRU, LSTM, Dense, TimeDistributed, Activation, Bidirectional, RepeatVector, Flatten, Permute, \
    Dropout, Lambda, Reshape, Embedding, Input, UpSampling1D
from keras.layers import Convolution2D as Conv2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import Convolution1D as Conv1D
from keras.layers import GlobalMaxPooling1D, MaxPooling1D
from keras.layers import BatchNormalization
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_fscore_support
import tensorflow as tf
from keras import backend as K


# def multitask_loss(y_true, y_pred):
#     # Avoid divide by 0
#     y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
#     # Multi-task loss
#     return K.mean(K.sum(- y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred), axis=1))

def build_multi(first):
    x = Input((None, 1))
    ev_in = Sequential()(x)
    ev = Conv1D(first, 16, activation='relu')(ev_in)
    ev = MaxPooling1D(16)(ev)
    ev = GRU(first*2, return_sequences=False)(ev)
    ev_out = Dense(1, activation='sigmoid', name='ev_out')(ev)

    ad_in = Sequential()(x)
    ad = Conv1D(first, kernel_size=4, padding="same", kernel_initializer='ones')(ad_in)
    ad = BatchNormalization(momentum=0.8)(ad)
    ad = Activation("relu")(ad)
    ad = UpSampling1D()(ad)
    ad = Conv1D(first, kernel_size=4, padding="same")(ad)
    ad = BatchNormalization(momentum=0.8)(ad)
    ad = Activation("relu")(ad)
    ad = Conv1D(1, kernel_size=4, padding="same")(ad)
    ad = MaxPooling1D(padding="same", pool_size=2)(ad)
    ad = GRU(first * 2, return_sequences=True)(ad)
    ad_out = TimeDistributed(Dense(1, activation='sigmoid'), name='ad_out')(ad)

    multi = Model(inputs=x, outputs=[ad_out, ev_out])
    multi.compile(optimizer='adam',
                  loss={'ad_out': 'binary_crossentropy', 'ev_out': 'mean_squared_error'},
                  loss_weights={'ad_out': 0.5, 'ev_out': 0.5})
    print(multi.summary())
    return multi