import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
from librosa import display
import librosa.display
from scipy import fft
from scipy.signal import butter, lfilter


def loadFilePickle(data_dir, filename):
    """
    Initializes a data generator object
      :param data_dir: the directory in which data are stored
      :param output_size: output size
      :param batch_size: The size of each batch returned by __getitem__
    """
    file_data = open(os.path.normpath('/'.join([data_dir, filename])), 'rb')
    Z = pickle.load(file_data)
    return Z


def plotTime(x, fs):
    """
    Initializes a data generator object
      :param data_dir: the directory in which data are stored
      :param output_size: output size
      :param batch_size: The size of each batch returned by __getitem__
    """
    fig, ax = plt.subplots(nrows=1, ncols=1)
    display.waveshow(x, sr=fs, ax=ax)


def plotFreq(x, fs, N):
    """
    Initializes a data generator object
      :param data_dir: the directory in which data are stored
      :param output_size: output size
      :param batch_size: The size of each batch returned by __getitem__
    """

    FFT = np.abs(fft.fftshift(fft.fft(x, n=N))[N // 2:])/len(N)
    freqs = fft.fftshift(fft.fftfreq(N) * fs)
    freqs = freqs[N // 2:]

    fig, ax = plt.subplots(1, 1)
    ax.semilogx(freqs, 20 * np.log10(np.abs(FFT)+1))
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Magnitude (dB)')
    ax.axis(xmin=20, xmax=22050)


def plotSpectogram(x, fs, N):
    """
    Initializes a data generator object
      :param data_dir: the directory in which data are stored
      :param output_size: output size
      :param batch_size: The size of each batch returned by __getitem__
    """

    D = librosa.stft(x, n_fft=N)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    fig, ax = plt.subplots()
    librosa.display.specshow(
        S_db, sr=fs, y_axis='linear', x_axis='time', ax=ax[0])
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [sec]')
    ax.label_outer()


def butter_lowpass(cutoff, fs, order=2):
    """
    Initializes a data generator object
      :param data_dir: the directory in which data are stored
      :param output_size: output size
      :param batch_size: The size of each batch returned by __getitem__
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_highpass(cutoff, fs, order=2):
    """
    Initializes a data generator object
      :param data_dir: the directory in which data are stored
      :param output_size: output size
      :param batch_size: The size of each batch returned by __getitem__
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def filterAudio(x, fs=48000, f_min=30, f_max=20000):
    """
    Initializes a data generator object
      :param data_dir: the directory in which data are stored
      :param output_size: output size
      :param batch_size: The size of each batch returned by __getitem__
    """
    [b, a] = butter_highpass(f_min, fs, order=2)
    [b2, a2] = butter_lowpass(f_max, fs, order=2)
    x = lfilter(b, a, x)
    x = lfilter(b2, a2, x)
    return np.array(x, dtype=np.float32)
