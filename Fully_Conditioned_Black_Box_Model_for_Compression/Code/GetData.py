import pickle
import os
import numpy as np


def get_data(data_dir, window, seed=422):
    os.environ['PYTHONHASHSEED'] = str(seed)
    file_data = open(os.path.normpath('/'.join([data_dir, 'TubeTech_train.pickle'])), 'rb')
    Z = pickle.load(file_data)
    x = np.array(Z['input'], dtype=np.float32)
    y = np.array(Z['target'], dtype=np.float32)
    z = np.array(Z['cond'], dtype=np.float32)

    sep = x.shape[0]
    sep_v = (sep * 20) // 100

    x_val = x[-sep_v:]
    y_val = y[-sep_v:]
    z_val = z[-sep_v:]

    x = x[:-sep_v]
    y = y[:-sep_v]
    z = z[:-sep_v]

    del Z

    step = window
    window = 2 * window

    all_inp, all_tar, all_cond = [], [], []
    length = x.shape[1]
    for i in range(x.shape[0]):
        for t in range(0, length - window, step):
            all_inp.append((np.array(x[i, t:t + window])).T)
            all_tar.append((np.array(y[i, t + step:t + window])).T)
            all_cond.append((np.array([z[i, 0], z[i, 1], z[i, 2], z[i, 3]])).T)

    all_inp = np.array(all_inp, dtype=np.float32)
    all_tar = np.array(all_tar, dtype=np.float32)
    all_cond = np.array(all_cond, dtype=np.float32)
    del x, z, y

    length = x_val.shape[1]
    all_inp_val, all_tar_val, all_cond_val = [], [], []
    for i in range(x_val.shape[0]):
        for t in range(0, length - window, step):
            all_inp_val.append((np.array(x_val[i, t:t + window])).T)
            all_tar_val.append((np.array(y_val[i, t + step:t + window])).T)
            all_cond_val.append((np.array([z_val[i, 0], z_val[i, 1], z_val[i, 2], z_val[i, 3]])).T)

    del x_val, z_val, y_val
    all_inp_val = np.array(all_inp_val, dtype=np.float32)
    all_tar_val = np.array(all_tar_val, dtype=np.float32)
    all_cond_val = np.array(all_cond_val, dtype=np.float32)

    return all_inp, all_tar, all_cond, all_inp_val, all_tar_val, all_cond_val


def get_test_data(data_dir, window, seed=422):
    os.environ['PYTHONHASHSEED'] = str(seed)

    file_data = open(os.path.normpath('/'.join([data_dir, 'TubeTech_test.pickle'])), 'rb')
    Z = pickle.load(file_data)
    x = np.array(Z['input'], dtype=np.float32)
    y = np.array(Z['target'], dtype=np.float32)
    z = np.array(Z['cond'], dtype=np.float32)

    window = 2 * window
    step = window // 2

    all_inp, all_tar, all_cond = [], [], []
    length = x.shape[1]
    n_examples = x.shape[0]
    for i in range(n_examples):
        for t in range(0, length - window, step):
            all_inp.append((np.array(x[i, t:t + window])).T)
            all_tar.append((np.array(y[i, t + step:t + window])).T)
            all_cond.append((np.array([z[i, 0], z[i, 1], z[i, 2], z[i, 3]])).T)

    del x, z, y
    all_inp = np.array(all_inp, dtype=np.float32)
    all_tar = np.array(all_tar, dtype=np.float32)
    all_cond = np.array(all_cond, dtype=np.float32)
    return all_inp, all_tar, all_cond