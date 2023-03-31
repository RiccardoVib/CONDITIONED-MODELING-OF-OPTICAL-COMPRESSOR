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


import tensorflow as tf
from tensorflow.keras import backend as K

def time_loss(y_true, y_pred):
    return tf.keras.metrics.mean_squared_error(y_true, y_pred)

def freq_loss(y_true, y_pred):
    return STFT_loss(y_true, y_pred)

def ESR_loss(y_true, y_pred):
    return tf.divide(K.sum(K.square(y_pred - y_true)), K.sum(K.square(y_true)))

def DCLoss(y_true, y_pred):
    return tf.divide(K.square(tf.divide(K.sum(y_pred - y_true), tf.cast(len(y_true), dtype=tf.float32))), tf.divide(K.sum(K.square(y_true)), tf.cast(len(y_true), dtype=tf.float32)))


def STFT_loss(y_true, y_pred):
    m = [8, 16, 32]
    loss = 0
    for i in range(len(m)):
        Y_true = tf.signal.stft(y_true, frame_length=m[i], frame_step=m[i] // 4)
        Y_pred = tf.signal.stft(y_pred, frame_length=m[i], frame_step=m[i] // 4)
        Y_true = K.pow(K.abs(Y_true), 2)
        Y_pred = K.pow(K.abs(Y_pred), 2)

        l_true = K.log(Y_true + 1)
        l_pred = K.log(Y_pred + 1)

        loss += tf.norm((l_true - l_pred), ord=1) + tf.norm((Y_true - Y_pred), ord=1)
    return loss
