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

        # w = 2  # n of column
        # h = len(all_inp)  # n of row
        # matrix = [[0 for x in range(w)] for y in range(h)]
        # for i in range(h):
        #     matrix[i][0] = all_inp[i]
        #     matrix[i][1] = all_tar[i]
        #
        # N = all_inp.shape[0]
        # n_train = N // 100 * 85
        # n_val = (N - n_train)
        #
        # for n in range(n_train):
        #     x.append(matrix[n][0])
        #     y.append(matrix[n][1])
        #
        # for n in range(n_val):
        #     x_val.append(matrix[n_train + n][0])
        #     y_val.append(matrix[n_train + n][1])
        #


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

        for t in range(inp.shape[1] // window):
            inp_temp = np.array(
                [inp[0, t : t + window], np.repeat(ratios[0], window), np.repeat(thresholds[0], window)])
            all_inp.append(inp_temp.T)
            tar_temp = np.array(tar[0, t : t + window])
            all_tar.append(tar_temp.T)

        all_inp = np.array(all_inp)
        all_tar = np.array(all_tar)

        N = all_inp.shape[0]
        for n in range(N):
            x_test.append(all_inp[n][0])
            y_test.append(all_tar[n][1])

        x_test = np.array(x_test)
        y_test = np.array(y_test)

    return x, y, x_val, y_val, x_test, y_test, scaler, fs