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
import glob
import pickle
import os
import numpy as np
from scipy.io import wavfile
from scipy.signal.windows import tukey
from tensorflow.keras.utils import Sequence
from audio_format import pcm2float
import librosa
import matplotlib.pyplot as plt
from Utils import filterAudio

class DataGeneratorCL1B(Sequence):

    def __init__(self, filename, data_dir, input_enc_size, input_dec_size, output_size, cond_size, set='train', shuffle=False,
                 batch_size=10):
        """
        Initializes a data generator object
          :param filename: name of the dataset
          :param data_dir: the directory in which data are stored
          :param input_enc_size: encoder input size
          :param input_dec_size: decoder input size
          :param output_size: output size
          :param cond_size: number of conditioning parameters
          :param shuffle: shuffle the data after each epoch
          :param batch_size: The size of each batch returned by __getitem__
        """
        file_data = open(os.path.normpath('/'.join([data_dir, filename])), 'rb')
        Z = pickle.load(file_data)
        x = np.array(Z['input'][:], dtype=np.float32)
        y = np.array(Z['target'][:], dtype=np.float32)
        z = np.array(Z['cond'][:], dtype=np.float32)

        x = x * np.array(tukey(x.shape[1], alpha=0.000005), dtype=np.float32).reshape(1, -1)
        y = y * np.array(tukey(x.shape[1], alpha=0.000005), dtype=np.float32).reshape(1, -1)
        #
        self.x = x
        self.y = y
        self.z = z
        self.data_dir = data_dir
        self.input_enc_size = input_enc_size
        self.input_dec_size = input_dec_size
        self.output_size = output_size
        self.cond_size = cond_size
        self.w = input_enc_size + input_dec_size
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indices = np.arange(self.x.shape[0])
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int((self.x.shape[0]) / self.batch_size)

    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)
            if i == self.__len__() - 1:
                self.on_epoch_end()

    def __getitem__(self, idx):
        ## Initializing Batch
        X = []  # np.empty((self.batch_size, 2*self.w))
        Y = []  # np.empty((self.batch_size, self.output_size))
        Z = []  # np.empty((self.batch_size, self.cond_size))

        step = self.output_size
        window = self.w
        lag = window - step

        # get the indices of the requested batch
        indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        length = self.x.shape[1]
        for i in indices:
            for t in range(0, length - window, step):
                X.append(np.array(self.x[i, t:t + window]).T)
                Y.append(np.array(self.y[i, t + lag:t + window]).T)
                Z.append(np.array([self.z[i, 0], self.z[i, 1], self.z[i, 2], self.z[i, 3]]).T)

        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.float32)
        Z = np.array(Z, dtype=np.float32)
        return [Z, X[:, :step], X[:, step:]], Y


class DataGeneratorLA2A(Sequence):

    def __init__(self, filename, data_dir, input_enc_size, input_dec_size, output_size, cond_size, shuffle=False, set='train',
                 batch_size=10):
        """
        Initializes a data generator object
          :param filename: name of the dataset
          :param data_dir: the directory in which data are stored
          :param input_enc_size: encoder input size
          :param input_dec_size: decoder input size
          :param output_size: output size
          :param cond_size: number of conditioning parameters
          :param shuffle: shuffle the data after each epoch
          :param batch_size: The size of each batch returned by __getitem__
        """
        file_data = open(os.path.normpath('/'.join([data_dir, filename])), 'rb')
        Z = pickle.load(file_data)
        if set == 'train':
            x = np.array(Z[0]['input'], dtype=np.float32)
            y = np.array(Z[0]['target'], dtype=np.float32)
            z1 = np.array(Z[0]['switch'], dtype=np.float32)
            z2 = np.array(Z[0]['peak'], dtype=np.float32)
        elif set == 'val':
            x = np.array(Z[1]['input'], dtype=np.float32)
            y = np.array(Z[1]['target'], dtype=np.float32)
            z1 = np.array(Z[1]['switch'], dtype=np.float32)
            z2 = np.array(Z[1]['peak'], dtype=np.float32)
        else:
            x = np.array(Z['input'], dtype=np.float32)
            y = np.array(Z['target'], dtype=np.float32)
            z1 = np.array(Z['switch'], dtype=np.float32)
            z2 = np.array(Z['peak'], dtype=np.float32)
        del Z

        x = x * np.array(tukey(x.shape[1], alpha=0.000005), dtype=np.float32).reshape(1, -1)
        y = y * np.array(tukey(x.shape[1], alpha=0.000005), dtype=np.float32).reshape(1, -1)

        length = x.shape[1]

        samples10 = 441000
        sec10 = length // samples10
        lim = sec10 * samples10

        x = x[:, :lim]
        y = y[:, :lim]

        z1 = np.repeat(z1, lim).reshape(-1, 1)
        z2 = np.repeat(z2/100, lim).reshape(-1, 1)

        self.z = np.concatenate((z1, z2), axis=-1)

        self.x = x.reshape(-1, samples10)
        self.y = y.reshape(-1, samples10)

        self.data_dir = data_dir
        self.input_enc_size = input_enc_size
        self.input_dec_size = input_dec_size
        self.output_size = output_size
        self.cond_size = cond_size
        self.w = input_enc_size + input_dec_size
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indices = np.arange(self.x.shape[0])
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int((self.x.shape[0]) / self.batch_size)

    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)
            if i == self.__len__() - 1:
                self.on_epoch_end()

    def __getitem__(self, idx):
        ## Initializing Batch
        X = []  # np.empty((self.batch_size, 2*self.w))
        Y = []  # np.empty((self.batch_size, self.output_size))
        Z = []  # np.empty((self.batch_size, self.cond_size))

        step = self.output_size
        window = self.w
        lag = window - step

        # get the indices of the requested batch
        indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        length = self.x.shape[1]

        for i in indices:
            for t in range(0, length - window, step):
                X.append(self.x[i, t:t + window].T)
                Y.append(self.y[i, t + lag:t + window].T)
                Z.append(self.z[i].T)

        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.float32)
        Z = np.array(Z, dtype=np.float32)

        return [Z, X[:, :step], X[:, step:]], Y
