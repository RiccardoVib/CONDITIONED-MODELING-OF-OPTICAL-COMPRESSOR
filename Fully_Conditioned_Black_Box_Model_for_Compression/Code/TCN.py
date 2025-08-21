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


class film(tf.keras.layers.Layer):
    def __init__(self, n_features):
        super(film, self).__init__()
        self.n_features = n_features
        self.bn = tf.keras.layers.BatchNormalization()
        self.adaptor = tf.keras.layers.Dense(n_features*2)

    def call(self, x, cond):
        cond = self.adaptor(cond)
        g, b = tf.split(cond, 2, axis=-1)
        g = tf.expand_dims(g, axis=1)
        b = tf.expand_dims(b, axis=1)
        x = self.bn(x)
        x = (x*g)+b
        return x


class TCNBlock(tf.keras.layers.Layer):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding="casual", dilation=1, grouped=False, conditional=False):
        super(TCNBlock, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self.grouped = grouped
        self.conditional = conditional

        groups = out_ch if grouped and (in_ch % out_ch == 0) else 1

        self.conv1 = tf.keras.layers.Conv1D(out_ch, kernel_size=kernel_size, padding='valid',
                                            dilation_rate=dilation, groups=groups, use_bias=False)
        if grouped:
            self.conv1b = tf.keras.layers.Conv1D(out_ch, kernel_size=1)
        if conditional:
            self.film = film(32)
        else:
            self.bn = tf.keras.layers.BatchNormalization(out_ch)
        self.relu = tf.keras.layers.PReLU()
        self.res = tf.keras.layers.Conv1D(out_ch,
                                   kernel_size=1,
                                   groups=groups,
                                   use_bias=False)
    def call(self, x, p, n):
        x_in = x
        x = self.conv1(x)
        if self.conditional:
            x = self.film(x, p)
        x = self.relu(x)
        x_res = self.res(x_in)

        # 72000, 71988, 71868, 70668, 58668
        len = int(1.2*10**(n+1))-1
        x = x + x_res[:, len:-1]

        return x

def create_tcn(nparams, n_inps, nblocks, kernel_size,
               dilation_growth=1, channel_growth=1, channel_width=32,
               grouped=False, conditional=False):
    inp = tf.keras.layers.Input(shape=(n_inps, 1), name='inputs')
    x = inp
    if nparams > 0:
        inp_p = tf.keras.layers.Input(shape=(nparams), name='params')
        p = tf.keras.layers.Dense(16)(inp_p)
        p = tf.keras.layers.ReLU()(p)
        p = tf.keras.layers.Dense(32)(p)
        p = tf.keras.layers.ReLU()(p)

    for n in range(nblocks):
        in_ch = out_ch if n > 0 else n_inps

        if channel_growth > 1:
            out_ch = channel_width
        else:
            out_ch = channel_width
        dilation = dilation_growth ** n

        x = TCNBlock(in_ch, out_ch, kernel_size, padding='same', dilation=dilation, grouped=grouped, conditional=conditional)(x, p, n)

    x = tf.keras.layers.Conv1D(1, kernel_size=1)(x)

    out = tf.nn.tanh(x)

    model = tf.keras.models.Model(inputs=[inp_p, inp], outputs=out, name='TCN')
    model.summary()

    return model


def create_tcn_nocond(n_inps, nblocks, kernel_size,
               dilation_growth=1, channel_growth=1, channel_width=32,
               grouped=False):
    inp = tf.keras.layers.Input(shape=(n_inps, 1), name='inputs')
    x = inp

    for n in range(nblocks):
        in_ch = out_ch if n > 0 else n_inps

        if channel_growth > 1:
            out_ch = channel_width
        else:
            out_ch = channel_width
        dilation = dilation_growth ** n

        x = TCNBlock(in_ch, out_ch, kernel_size, padding='same', dilation=dilation, grouped=grouped, conditional=False)(x, 0., n)

    x = tf.keras.layers.Conv1D(1, kernel_size=1)(x)

    out = tf.nn.tanh(x)

    model = tf.keras.models.Model(inputs=inp, outputs=out, name='TCN')
    model.summary()

    return model
