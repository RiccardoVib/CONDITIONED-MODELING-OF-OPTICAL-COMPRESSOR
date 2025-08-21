# Copyright (C) 2023 Riccardo Simionato, University of Oslo
# Inquiries: riccardo.simionato.vib@gmail.com.com
#
# This code is free software: you can redistribute it and/or modify it under the terms
# of the GNU Lesser General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
#
# This code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Less General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along with this code.
# If not, see <http://www.gnu.org/licenses/>.
#
# If you use this code or any part of it in any program or publication, please acknowledge
# its authors by adding a reference to this publication:
#
# R. Simionato, 2023, "Fully Conditioned Black Box Model for Compression" in proceedings of the 23th Digital Audio Effect Conference, Copenaghen, Denmark.

from tensorflow.keras.layers import Input, Dense, LSTM, Add, Conv1D, BatchNormalization, PReLU, Multiply, ReLU
from tensorflow.keras.models import Model
import tensorflow as tf

def create_model_ED_CNN(cond_dim, input_dim, units, activation='sigmoid'):
    D = cond_dim
    T = input_dim//2

    cond_inputs = Input(shape=(D,), name='enc_cond')
    encoder_inputs = Input(shape=(T, 1), name='enc_input')
    decoder_inputs = Input(shape=(T, 1), name='dec_input')

    cond_dense_h = Dense(units, name='Dense_cond_h')(cond_inputs)
    cond_dense_c = Dense(units, name='Dense_cond_c')(cond_inputs)

    state_h = Conv1D(units, T, name='Conv_h')(encoder_inputs)
    state_c = Conv1D(units, T, name='Conv_c')(encoder_inputs)

    states_h = Add()([state_h[:, 0, :], cond_dense_h])
    states_c = Add()([state_c[:, 0, :], cond_dense_c])
    encoder_states = [states_h, states_c]

    decoder_outputs = LSTM(units,  return_sequences=False, name='LSTM_De')(decoder_inputs, initial_state=encoder_states)

    decoder_outputs = Dense(units, activation=activation, name='DenseLay')(decoder_outputs)
    decoder_outputs = Dense(T, name='OutLay')(decoder_outputs)

    model = Model([cond_inputs, encoder_inputs, decoder_inputs], decoder_outputs)
    model.summary()
    return model

#
# def create_model_TCN(cond_dim, input_dim, out, units, ker, dilation=2):
#     T = input_dim  # time window
#     D = cond_dim  # features
#
#     cond_inputs = Input(shape=(D,), name='cond')
#     cond_dense = Dense(16, name='Dense_cond_1')(cond_inputs)
#     cond_dense = ReLU(name='relu')(cond_dense)
#     cond_dense = Dense(32, name='Dense_cond_2')(cond_dense)
#     cond_dense = ReLU(name='relu2')(cond_dense)
#     cond_dense = Dense(32, name='Dense_cond_3')(cond_dense)
#     cond_dense = ReLU(name='relu3')(cond_dense)
#
#     inputs = Input(shape=(T, 1), name='inp')
#     # TCN block
#     d = int(np.power(dilation, 0))
#     outputs = Conv1D(units, ker, dilation_rate=d, name='Conv_dec_1')(inputs)
#     outputs = BatchNormalization(name='BN_1')(outputs)
#     # FiLM
#     C = d * (ker-1)
#     film = Dense((T - C) * 2)(cond_dense)
#     g, b = tf.split(film, 2, axis=-1)
#     outputs = Multiply()([outputs, tf.expand_dims(g, axis=-1)])
#     outputs = Add()([outputs, tf.expand_dims(b, axis=-1)])
#     res = Conv1D(1, 1, name='Conv_res_1')(inputs)
#     outputs = ReLU()(outputs + res[:, C:, :])
#
#     d = int(np.power(dilation, 1))
#     C = C + d * (ker - 1)
#     outputs = Conv1D(units, ker, dilation_rate=d, name='Conv_dec_2')(outputs)
#     outputs = BatchNormalization(name='BN_2')(outputs)
#     # FiLM
#     film = Dense((T - C)*2)(cond_dense)
#     g, b = tf.split(film, 2, axis=-1)
#     outputs = Multiply()([outputs, tf.expand_dims(g, axis=-1)])
#     outputs = Add()([outputs, tf.expand_dims(b, axis=-1)])
#     res = Conv1D(1, 1, name='Conv_res_2')(inputs)
#     outputs = ReLU()(outputs + res[:, C:, :])
#
#     d = int(np.power(dilation, 2))
#     C = C + d * (ker - 1)
#     outputs = Conv1D(units, ker, dilation_rate=d, name='Conv_dec_3')(outputs)
#     outputs = BatchNormalization(name='BN_3')(outputs)
#     # FiLM
#     film = Dense((T - C) * 2)(cond_dense)
#     g, b = tf.split(film, 2, axis=-1)
#     outputs = Multiply()([outputs, tf.expand_dims(g, axis=-1)])
#     outputs = Add()([outputs, tf.expand_dims(b, axis=-1)])
#     res = Conv1D(1, 1, name='Conv_res_3')(inputs)
#     outputs = ReLU()(outputs + res[:, C:, :])
#
#     d = int(np.power(dilation, 3))
#     C = C + d * (ker - 1)
#     outputs = Conv1D(units, ker, dilation_rate=d, name='Conv_dec_4')(outputs)
#     outputs = BatchNormalization(name='BN_4')(outputs)
#     # FiLM
#     film = Dense((T - C) * 2)(cond_dense)
#     g, b = tf.split(film, 2, axis=-1)
#     outputs = Multiply()([outputs, tf.expand_dims(g, axis=-1)])
#     outputs = Add()([outputs, tf.expand_dims(b, axis=-1)])
#     res = Conv1D(1, 1, name='Conv_res_4')(inputs)
#     outputs = ReLU()(outputs + res[:, C:, :])
#
#     outputs = Conv1D(out, 1)(outputs)
#
#     model = Model([cond_inputs, inputs], outputs)
#     model.summary()
#     return model

