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

    RMS = []
    for j in range(Y.shape[0]):
        _rms = []
        for i in range(Y.shape[1] // step):
            _rms.append(tf.sqrt(tf.reduce_mean(tf.square(tf.constant(Y[j])))))
        RMS.append(_rms)
    RMS = np.array(RMS)
    #
    # STFT = tf.abs(tf.signal.stft(Y, fft_length=512, frame_length=512, frame_step=512 // 4, pad_end=True))
    # STFT = tf.pow(tf.abs(STFT), 2)
    # STFT = tf.math.log(STFT + 1)

    return X, Y, Z, RMS


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

    # RMS = []
    # for j in range(Y.shape[0]):
    #     _rms = []
    #     for i in range(Y.shape[1] // step):
    #         _rms.append(tf.sqrt(tf.reduce_mean(tf.square(tf.constant(Y[j])))))
    #     RMS.append(_rms)
    # RMS = np.array(RMS)
    #
    # STFT = tf.abs(tf.signal.stft(Y, fft_length=512, frame_length=512, frame_step=512 // 4, pad_end=True))
    # STFT = tf.pow(tf.abs(STFT), 2)
    # STFT = tf.math.log(STFT + 1)

    return X, Y, Z#, RMS, STFT



def get_test_data_chorus(data_dir, w, o, seed=422):
    os.environ['PYTHONHASHSEED'] = str(seed)

    file_data = open(os.path.normpath('/'.join([data_dir, 'LA2A_test_data.pickle'])), 'rb')

    Z = pickle.load(file_data)
    x = np.array(Z['x'][:, :44100*5], dtype=np.float32)
    y = np.array(Z['y'][:, :44100*5], dtype=np.float32)
    z = np.array(Z['z'], dtype=np.float32)

    X = []  # np.empty((self.batch_size, 2*self.w))
    Y = []  # np.empty((self.batch_size, self.output_size))
    Z = []  # np.empty((self.batch_size, self.cond_size))

    step = o
    window = 2 * w
    lag = window - step

    length = x.shape[1]
    for i in range(x.shape[0]):
        for t in range(0, length - window, step):
            X.append(np.array(x[i, t:t + window]).T)
            Y.append(np.array(y[i, t + lag:t + window]).T)
            Z.append(np.array([z[i, 0], z[i, 1]]).T)

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    Z = np.array(Z, dtype=np.float32)

    #RMS = []
    #for j in range(Y.shape[0]):
    #    _rms = []
    #    for i in range(Y.shape[1] // step):
    #        _rms.append(tf.sqrt(tf.reduce_mean(tf.square(tf.constant(Y[j])))))
    #    RMS.append(_rms)
    #RMS = np.array(RMS)

    #STFT = tf.abs(tf.signal.stft(Y, fft_length=512, frame_length=512, frame_step=512 // 4, pad_end=True))
    #STFT = tf.pow(tf.abs(STFT), 2)
    #STFT = tf.math.log(STFT + 1)

    return X, Y, Z#, RMS, STFT

if __name__ == '__main__':
    data_dir = '../../Files'

