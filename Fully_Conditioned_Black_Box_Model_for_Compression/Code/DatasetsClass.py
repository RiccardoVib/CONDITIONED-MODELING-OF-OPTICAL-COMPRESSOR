import pickle
import os
import numpy as np
from tensorflow.keras.utils import Sequence
import tensorflow as tf

class DataGeneratorCL1B(Sequence):

    def __init__(self, filename, data_dir, input_enc_size, input_dec_size, output_size, cond_size, window, shuffle=False, batch_size=10):
        """
        Initializes a data generator object
          :param data_dir: the directory in which data are stored
          :param output_size: output size
          :param shuffle: shuffle the data after each epoch
          :param batch_size: The size of each batch returned by __getitem__
        """
        file_data = open(os.path.normpath('/'.join([data_dir, filename])), 'rb')
        Z = pickle.load(file_data)
        x = np.array(Z['x'], dtype=np.float32)
        y = np.array(Z['y'], dtype=np.float32)
        z = np.array(Z['z'], dtype=np.float32)

        self.x = x
        self.y = y
        self.z = z
        self.data_dir = data_dir
        self.input_enc_size = input_enc_size
        self.input_dec_size = input_dec_size
        self.output_size = output_size
        self.cond_size = cond_size
        self.w = window
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
        X = []#np.empty((self.batch_size, 2*self.w))
        Y = []#np.empty((self.batch_size, self.output_size))
        Z = []#np.empty((self.batch_size, self.cond_size))

        step = self.output_size
        window = 2 * self.w
        lag = window - step

        # get the indices of the requested batch
        indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]

        length = self.x.shape[1]
        for i in indices:
            for t in range(0, length - window, step):
                X.append(np.array(self.x[i, t:t + window]).T)
                Y.append(np.array(self.y[i, t + lag:t + window]).T)
                Z.append(np.array([self.z[i, 0], self.z[i, 1], self.z[i, 2], self.z[i, 3]]).T)

        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.float32)
        Z = np.array(Z, dtype=np.float32)

        #RMS = []
        #for j in range(Y.shape[0]):
        #    _rms = tf.sqrt(tf.reduce_mean(tf.square(tf.constant(Y[j]))))
        #    RMS.append(_rms)
        #RMS = np.array(RMS)
        #
        # STFT = tf.abs(tf.signal.stft(Y, fft_length=512, frame_length=512, frame_step=512 // 4, pad_end=True))
        # STFT = tf.pow(tf.abs(STFT), 2)
        # STFT = tf.math.log(STFT + 1)

        return [Z, X[:, :self.w], X[:, self.w:]], Y
    
class DataGeneratorLA2A(Sequence):

    def __init__(self, filename, data_dir, input_enc_size, input_dec_size, output_size, cond_size, window, shuffle=False, batch_size=10):
        """
        Initializes a data generator object
          :param data_dir: the directory in which data are stored
          :param output_size: output size
          :param shuffle: shuffle the data after each epoch
          :param batch_size: The size of each batch returned by __getitem__
        """
        file_data = open(os.path.normpath('/'.join([data_dir, filename])), 'rb')
        Z = pickle.load(file_data)
        x = np.array(Z['x'], dtype=np.float32)
        y = np.array(Z['y'], dtype=np.float32)
        z = np.array(Z['z'], dtype=np.float32)

        length = x.shape[1]

        samples10 = 441000
        sec10 = length//samples10
        lim = sec10*samples10

        x = x[:, :lim]
        y = y[:, :lim]

        self.x = x.reshape(-1, samples10)
        self.y = y.reshape(-1, samples10)

        z0 = np.repeat(z[:, 0], self.x.shape[0]//3).reshape(-1, 1)
        z1 = np.repeat(z[:, 1], self.x.shape[0]//3).reshape(-1, 1)

        self.z = np.concatenate((z0, z1), axis=-1)

        self.data_dir = data_dir
        self.input_enc_size = input_enc_size
        self.input_dec_size = input_dec_size
        self.output_size = output_size
        self.cond_size = cond_size
        self.w = window
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
        X = []#np.empty((self.batch_size, 2*self.w))
        Y = []#np.empty((self.batch_size, self.output_size))
        Z = []#np.empty((self.batch_size, self.cond_size))

        step = self.output_size
        window = 2 * self.w
        lag = window - step

        # get the indices of the requested batch
        indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]

        length = self.x.shape[1]

        for i in indices:
            for t in range(0, length - window, step):
                X.append(np.array(self.x[i, t:t + window]).T)
                Y.append(np.array(self.y[i, t + lag:t + window]).T)
                Z.append(np.array([self.z[i, 0], self.z[i, 1]]).T)

        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.float32)
        Z = np.array(Z, dtype=np.float32)

        RMS = []
        for j in range(Y.shape[0]):
            _rms = tf.sqrt(tf.reduce_mean(tf.square(tf.constant(Y[j]))))
            RMS.append(_rms)
        RMS = np.array(RMS)

        return [Z, X[:, :self.w], X[:, self.w:]], [Y, RMS]


class DataGeneratorRev(Sequence):

    def __init__(self, filename, data_dir, input_enc_size, input_dec_size, output_size, cond_size, window,
                 shuffle=False, batch_size=10):
        """
        Initializes a data generator object
          :param data_dir: the directory in which data are stored
          :param output_size: output size
          :param shuffle: shuffle the data after each epoch
          :param batch_size: The size of each batch returned by __getitem__
        """
        file_data = open(os.path.normpath('/'.join([data_dir, filename])), 'rb')
        Z = pickle.load(file_data)
        self.x = np.array(Z['x'], dtype=np.float32)
        self.y = np.array(Z['y'], dtype=np.float32)
        self.z = np.array(Z['z'], dtype=np.float32)
        length = self.x.shape[1]

        samples1 = 44100
        sec1 = length // samples1
        lim = sec1 * samples1

        self.x = self.x[:, :lim]
        self.y = self.y[:, :lim]

        self.x = self.x.reshape(-1, samples1)
        self.y = self.y.reshape(-1, samples1)

        z0 = np.repeat(self.z, self.x.shape[0] // self.z.shape[0]).reshape(-1, 1)
        # z1 = np.repeat(self.z[:, 1], self.x.shape[0]//3).reshape(-1, 1)

        # self.z = np.concatenate((z0, z1), axis=-1)
        self.z = z0

        self.data_dir = data_dir
        self.input_enc_size = input_enc_size
        self.input_dec_size = input_dec_size
        self.output_size = output_size
        self.cond_size = cond_size
        self.w = window
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
        window = 2 * self.w
        lag = window - step

        # get the indices of the requested batch
        indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        length = self.x.shape[1]
        for i in indices:
            for t in range(0, length - window, step):
                X.append(np.array(self.x[i, t:t + window]).T)
                Y.append(np.array(self.y[i, t + lag:t + window]).T)
                # Z.append(np.array([self.z[i, 0], self.z[i, 1]]).T)
                Z.append(np.array(self.z[i, 0]).T)

        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.float32)
        Z = np.array(Z, dtype=np.float32)

        # RMS = []
        # for j in range(Y.shape[0]):
        #    _rms = []
        #    for i in range(Y.shape[1] // step):
        #        _rms.append(tf.sqrt(tf.reduce_mean(tf.square(tf.constant(Y[j])))))
        #    RMS.append(_rms)
        # RMS = np.array(RMS)

        # STFT = K.abs(tf.signal.stft(Y, fft_length=512, frame_length=512, frame_step=512 // 4, pad_end=True))
        # STFT = K.pow(K.abs(STFT), 2)
        # STFT = K.log(STFT + 1)

        # return [Z, X[:, :self.w], X[:, self.w:]], [Y, RMS, STFT]
        return [Z, X[:, :self.w], X[:, self.w:]], Y