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


def get_data(data_dir, window, seed=422):
    os.environ['PYTHONHASHSEED'] = str(seed)
    file_data = open(os.path.normpath('/'.join([data_dir, 'LA2A_train.pickle'])), 'rb')

    Z = pickle.load(file_data)
    z = []
    x = np.array(Z[0]['input'], dtype=np.float32)
    y = np.array(Z[0]['target'], dtype=np.float32)

    for i in range(x.shape[0]):
        z.append([Z[0]['switch'][i], int(Z[0]['peak'][i]) / 100])
    z = np.array(z, dtype=np.float32)

    z_v = []
    x_v = np.array(Z[1]['input'], dtype=np.float32)
    y_v = np.array(Z[1]['target'], dtype=np.float32)

    for i in range(x_v.shape[0]):
        z_v.append([Z[1]['switch'][i], int(Z[1]['peak'][i]) / 100])
    z_v = np.array(z_v, dtype=np.float32)

    del Z
    step = window
    window = 2 * window
    length = x.shape[1]
    all_inp, all_tar, all_cond = [], [], []

    for i in range(x.shape[0]):
        for t in range(0, length - window, step):
            inp_temp = np.array(x[i, t:t + window])
            all_inp.append(inp_temp.T)
            tar_temp = np.array(y[i, t+step:t + window])
            all_tar.append(tar_temp.T)
            cond_temp = np.array([z[i, 0], z[i, 1]])
            all_cond.append(cond_temp.T)

    all_inp = np.array(all_inp, dtype=np.float32)
    all_tar = np.array(all_tar, dtype=np.float32)
    all_cond = np.array(all_cond, dtype=np.float32)

    all_inp_val, all_tar_val, all_cond_val = [], [], []
    for i in range(x_v.shape[0]):
        for t in range(0, length - window, step):
            inp_temp = np.array(x_v[i, t:t + window])
            all_inp_val.append(inp_temp.T)
            tar_temp = np.array(y_v[i, t+step:t + window])
            all_tar_val.append(tar_temp.T)
            cond_temp = np.array([z_v[i, 0], z_v[i, 1]])
            all_cond_val.append(cond_temp.T)

    all_inp_val = np.array(all_inp_val, dtype=np.float32)
    all_tar_val = np.array(all_tar_val, dtype=np.float32)
    all_cond_val = np.array(all_cond_val, dtype=np.float32)
    return all_inp, all_tar, all_cond, all_inp_val, all_tar_val, all_cond_val


def get_data_test(data_dir, window, seed=422):
    os.environ['PYTHONHASHSEED'] = str(seed)
    file_data = open(os.path.normpath('/'.join([data_dir, 'LA2A_test.pickle'])), 'rb')

    Z = pickle.load(file_data)
    z = []
    x = np.array(Z['input'], dtype=np.float32)
    y = np.array(Z['target'], dtype=np.float32)
    for i in range(x.shape[0]):
        z.append([Z['switch'][i], int(Z['peak'][i]) / 100])
    z = np.array(z, dtype=np.float32)
    length = x.shape[1]
    del Z
    step = window
    window = 2 * window

    all_inp, all_tar, all_cond = [], [], []

    for i in range(x.shape[0]):
        for t in range(0, length - window, step):
            inp_temp = np.array(x[i, t:t + window])
            all_inp.append(inp_temp.T)
            tar_temp = np.array(y[i, t+step:t + window])
            all_tar.append(tar_temp.T)
            cond_temp = np.array([z[i, 0], z[i, 1]])
            all_cond.append(cond_temp.T)

    all_inp = np.array(all_inp, dtype=np.float32)
    all_tar = np.array(all_tar, dtype=np.float32)
    all_cond = np.array(all_cond, dtype=np.float32)

    return all_inp, all_tar, all_cond