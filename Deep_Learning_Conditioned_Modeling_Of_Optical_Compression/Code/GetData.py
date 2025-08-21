# Copyright (C) 2022 Riccardo Simionato, University of Oslo
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
# R. Simionato, 2022, "Deep Learning Conditioned Modeling of Optical Compression" in proceedings of the 22th Digital Audio Effect Conference, Vienna, Austria.



import pickle
import random
import os
import numpy as np
from Preprocess import my_scaler


def get_data(data_dir, window, inference, scaler, seed=422):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    x, y, x_val, y_val, x_test, y_test = [], [], [], [], [], []

    if not inference:
        # -----------------------------------------------------------------------------------------------------------------
        # Load data
        # -----------------------------------------------------------------------------------------------------------------
        meta = open(os.path.normpath('/'.join([data_dir, 'metadatas_train.pickle'])), 'rb')
        file_data = open(os.path.normpath('/'.join([data_dir, 'data_train.pickle'])), 'rb')

        Z = pickle.load(file_data)
        inp = Z['inp']
        tar = Z['tar']

        inp = np.array(inp, dtype=np.float32)
        tar = np.array(tar, dtype=np.float32)
        Z = [inp, tar]
        Z = np.array(Z)

        M = pickle.load(meta)
        ratios = M['ratio']
        threshold = M['threshold']
        fs = M['samplerate']

        # -----------------------------------------------------------------------------------------------------------------
        # Scale data to be within (0, 1)
        # -----------------------------------------------------------------------------------------------------------------

        scaler = my_scaler()
        scaler.fit(Z)

        inp = scaler.transform(inp)
        tar = scaler.transform(tar)

        ratios = np.array(ratios, dtype=np.float32)
        thresholds = np.array(threshold, dtype=np.float32)

        scaler_ratios = my_scaler()
        scaler_threshold = my_scaler()

        scaler_ratios.fit(ratios)
        scaler_threshold.fit(thresholds)
        thresholds = scaler_threshold.transform(thresholds)
        ratios = scaler_ratios.transform(ratios)

        scaler = [scaler, scaler_ratios, scaler_threshold]

        all_inp, all_tar = [], []

        for i in range(inp.shape[0]):
            for t in range(inp.shape[1] - window):
                inp_temp = np.array([inp[i, t : t + window], np.repeat(ratios[i], window),
                                     np.repeat(thresholds[i], window)])
                all_inp.append(inp_temp.T)
                tar_temp = np.array(tar[i, t : t + window])
                all_tar.append(tar_temp.T)

        all_inp = np.array(all_inp)
        all_tar = np.array(all_tar)

        N = all_inp.shape[0]
        n_train = N // 100 * 85
        n_val = (N - n_train)
        for n in range(n_train):
            x.append(all_inp[n])
            y.append(all_tar[n])

        for n in range(n_val):
            x_val.append(all_inp[n_train + n])
            y_val.append(all_tar[n_train + n])

        x = np.array(x)
        y = np.array(y)
        x_val = np.array(x_val)
        y_val = np.array(y_val)

    else:
        # inference
        all_inp, all_tar = [], []

        meta = open(os.path.normpath('/'.join([data_dir, 'metadata_test.pickle'])), 'rb')
        file_data = open(os.path.normpath('/'.join([data_dir, 'data_test.pickle'])), 'rb')
        Z = pickle.load(file_data)
        inp = Z['inp']
        tar = Z['tar']
        inp = np.array(inp, dtype=np.float32)
        tar = np.array(tar, dtype=np.float32)
        M = pickle.load(meta)
        ratios = M['ratio']
        threshold = M['threshold']
        inp = scaler[0].transform(inp)
        tar = scaler[0].transform(tar)
        ratios = np.array(ratios, dtype=np.float32)
        thresholds = np.array(threshold, dtype=np.float32)
        thresholds = scaler[2].transform(thresholds)
        ratios = scaler[1].transform(ratios)

        for t in range(inp.shape[1] - window):
            inp_temp = np.array(
                [inp[0, t : t + window], np.repeat(ratios[0], window), np.repeat(thresholds[0], window)])
            all_inp.append(inp_temp.T)
            tar_temp = np.array(tar[0, t : t + window])
            all_tar.append(tar_temp.T)

        all_inp = np.array(all_inp)
        all_tar = np.array(all_tar)

        N = all_inp.shape[0]
        for n in range(N):
            x_test.append(all_inp[n])
            y_test.append(all_tar[n])

        x_test = np.array(x_test)
        y_test = np.array(y_test)

    return x, y, x_val, y_val, x_test, y_test, scaler, fs
