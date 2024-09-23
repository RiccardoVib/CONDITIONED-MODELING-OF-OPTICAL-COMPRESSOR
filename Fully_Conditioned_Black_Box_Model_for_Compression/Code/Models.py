import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Add, Conv1D, BatchNormalization, PReLU, Multiply, ReLU
from tensorflow.keras.models import Model
import pickle
from scipy import signal


def create_model_ED_CNN(cond_dim, input_dim, units, activation='sigmoid', drop=0.):
    D = cond_dim
    T = input_dim
    cond_inputs = Input(shape=(D), name='enc_cond')
    encoder_inputs = Input(shape=(T, 1), name='enc_input')

    cond_dense_h = Dense(units, name='Dense_cond_h')(cond_inputs)
    cond_dense_c = Dense(units, name='Dense_cond_c')(cond_inputs)

    state_h = Conv1D(units, T, name='Conv_h')(encoder_inputs)
    state_c = Conv1D(units, T, name='Conv_c')(encoder_inputs)

    states_h = Add()([state_h[:, 0, :], cond_dense_h])
    states_c = Add()([state_c[:, 0, :], cond_dense_c])

    encoder_states = [states_h, states_c]

    decoder_inputs = Input(shape=(T, 1), name='dec_input')

    outputs = LSTM(units, return_sequences=False, return_state=False, name='LSTM_De',
                   dropout=drop)(decoder_inputs, initial_state=encoder_states)

    decoder_outputs = Dense(units, activation=activation, name='DenseLay')(outputs)

    decoder_outputs = Dense(T, name='OutLay')(decoder_outputs)

    model = Model([cond_inputs, encoder_inputs, decoder_inputs], decoder_outputs)
    model.summary()
    return model



def create_model_TCN(cond_dim, input_dim, out, units, ker, dilation=2):
    T = input_dim  # time window
    D = cond_dim  # features

    cond_inputs = Input(shape=(D), name='cond')
    cond_dense = Dense(16, name='Dense_cond_1')(cond_inputs)
    cond_dense = ReLU(name='relu')(cond_dense)
    cond_dense = Dense(32, name='Dense_cond_2')(cond_dense)
    cond_dense = ReLU(name='relu2')(cond_dense)
    cond_dense = Dense(32, name='Dense_cond_3')(cond_dense)
    cond_dense = ReLU(name='relu3')(cond_dense)

    inputs = Input(shape=(T, 1), name='inp')
    # TCN block
    d = int(np.power(dilation, 0))
    outputs = Conv1D(units, ker, dilation_rate=d, name='Conv_dec_1')(inputs)
    outputs = BatchNormalization(name='BN_1')(outputs)
    # FiLM
    C = d * (ker-1)
    film = Dense((T - C) * 2)(cond_dense)
    g, b = tf.split(film, 2, axis=-1)
    outputs = Multiply()([outputs, tf.expand_dims(g, axis=-1)])
    outputs = Add()([outputs, tf.expand_dims(b, axis=-1)])
    res = Conv1D(1, 1, name='Conv_res_1')(inputs)
    outputs = ReLU()(outputs + res[:, C:, :])

    d = int(np.power(dilation, 1))
    C = C + d * (ker - 1)
    outputs = Conv1D(units, ker, dilation_rate=d, name='Conv_dec_2')(outputs)
    outputs = BatchNormalization(name='BN_2')(outputs)
    # FiLM
    film = Dense((T - C)*2)(cond_dense)
    g, b = tf.split(film, 2, axis=-1)
    outputs = Multiply()([outputs, tf.expand_dims(g, axis=-1)])
    outputs = Add()([outputs, tf.expand_dims(b, axis=-1)])
    res = Conv1D(1, 1, name='Conv_res_2')(inputs)
    outputs = ReLU()(outputs + res[:, C:, :])

    d = int(np.power(dilation, 2))
    C = C + d * (ker - 1)
    outputs = Conv1D(units, ker, dilation_rate=d, name='Conv_dec_3')(outputs)
    outputs = BatchNormalization(name='BN_3')(outputs)
    # FiLM
    film = Dense((T - C) * 2)(cond_dense)
    g, b = tf.split(film, 2, axis=-1)
    outputs = Multiply()([outputs, tf.expand_dims(g, axis=-1)])
    outputs = Add()([outputs, tf.expand_dims(b, axis=-1)])
    res = Conv1D(1, 1, name='Conv_res_3')(inputs)
    outputs = ReLU()(outputs + res[:, C:, :])

    d = int(np.power(dilation, 3))
    C = C + d * (ker - 1)
    outputs = Conv1D(units, ker, dilation_rate=d, name='Conv_dec_4')(outputs)
    outputs = BatchNormalization(name='BN_4')(outputs)
    # FiLM
    film = Dense((T - C) * 2)(cond_dense)
    g, b = tf.split(film, 2, axis=-1)
    outputs = Multiply()([outputs, tf.expand_dims(g, axis=-1)])
    outputs = Add()([outputs, tf.expand_dims(b, axis=-1)])
    res = Conv1D(1, 1, name='Conv_res_4')(inputs)
    outputs = ReLU()(outputs + res[:, C:, :])

    outputs = Conv1D(out, 1)(outputs)

    model = Model([cond_inputs, inputs], outputs)
    model.summary()
    return model
