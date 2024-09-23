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


import pickle
import os
import numpy as np
import tensorflow as tf

def get_test_data(data_dir, w, output_size, seed=422):
    os.environ['PYTHONHASHSEED'] = str(seed)

    file_data = open(os.path.normpath('/'.join([data_dir, 'TubeTech_train_reduced_test.pickle'])), 'rb')

    Z = pickle.load(file_data)
    x = np.array(Z['x'], dtype=np.float32)
    y = np.array(Z['y'], dtype=np.float32)
    z = np.array(Z['z'], dtype=np.float32)

    X = []  # np.empty((self.batch_size, 2*self.w))
    Y = []  # np.empty((self.batch_size, self.output_size))
    Z = []  # np.empty((self.batch_size, self.cond_size))

    step = output_size
    window = 2 * w
    lag = window - step

    length = x.shape[1]
    for i in range(x.shape[0]):
        for t in range(0, length - window, step):
            X.append(np.array(x[i, t:t + window]).T)
            Y.append(np.array(y[i, t + lag:t + window]).T)
            Z.append(np.array([z[i, 0], z[i, 1], z[i, 2], z[i, 3]]).T)

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    Z = np.array(Z, dtype=np.float32)

    return X, Y, Z


def get_test_data_la2a(data_dir, w, output_size, seed=422):
    os.environ['PYTHONHASHSEED'] = str(seed)

    file_data = open(os.path.normpath('/'.join([data_dir, 'LA2A_test_data.pickle'])), 'rb')

    Z = pickle.load(file_data)
    x = np.array(Z['x'], dtype=np.float32)
    y = np.array(Z['y'], dtype=np.float32)
    z = np.array(Z['z'], dtype=np.float32)

    X = []  # np.empty((self.batch_size, 2*self.w))
    Y = []  # np.empty((self.batch_size, self.output_size))
    Z = []  # np.empty((self.batch_size, self.cond_size))

    step = output_size
    window = 2 * w
    lag = window - step

    length = x.shape[1]//2
    for i in range(x.shape[0]):
        for t in range(0, length - window, step):
            X.append(np.array(x[i, t:t + window]).T)
            Y.append(np.array(y[i, t + lag:t + window]).T)
            Z.append(np.array([z[i, 0], z[i, 1]]).T)

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    Z = np.array(Z, dtype=np.float32)

    return X, Y, Z
