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



import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
from librosa import display
from scipy.io import wavfile


def writeResults(test_loss, results, b_size, learning_rate, loss_type, model_save_dir, save_folder,
                 index):
    results = {
        'Test_Loss': test_loss,
        'Min_val_loss': np.min(results.history['val_loss']),
        'Min_train_loss': np.min(results.history['loss']),
        'b_size': b_size,
        'learning_rate': learning_rate,
        'loss_type': loss_type,
        # 'Train_loss': results.history['loss'],
        'Val_loss': results.history['val_loss']
    }
    with open(os.path.normpath('/'.join([model_save_dir, save_folder, 'results_' + str(index) + '.txt'])), 'w') as f:
        for key, value in results.items():
            print('\n', key, '  : ', value, file=f)
        pickle.dump(results,
                    open(os.path.normpath('/'.join([model_save_dir, save_folder, 'results_' + str(index) + '.pkl'])),
                         'wb'))


def plotResult(predictions, x, y, save_folder):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    # ax.plot(predictions, label='pred')
    # ax.plot(x, label='inp')
    # ax.plot(y, label='tar')
    display.waveshow(y, sr=44100, ax=ax, label='Target', alpha=0.9)
    display.waveshow(predictions, sr=44100, ax=ax, label='Prediction', alpha=0.7)
    # ax.label_outer()
    ax.legend(loc='upper right')
    fig.savefig('../../TrainedModels/' + save_folder + '/plot')
    plt.close('all')


def plotTraining(loss_training, loss_val, model_save_dir, save_folder):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(np.array(loss_training), label='train'),
    ax.plot(np.array(loss_val), label='validation')
    plt.ylabel('mse')
    plt.xlabel('epoch')
    plt.title('train vs. validation accuracy')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=False, ncol=2)
    fig.savefig(model_save_dir + '/' + save_folder + '/' + 'loss.png')
    plt.close('all')


def predictWaves(predictions, x_test, y_test, model_save_dir, save_folder, T):
    pred_name = '_pred.wav'
    inp_name = '_inp.wav'
    tar_name = '_tar.wav'

    pred_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', pred_name))
    inp_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', inp_name))
    tar_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', tar_name))

    if not os.path.exists(os.path.dirname(pred_dir)):
        os.makedirs(os.path.dirname(pred_dir))

    wavfile.write(pred_dir, 48000, predictions.reshape(-1))
    wavfile.write(inp_dir, 48000, x_test[:, T:].reshape(-1))
    wavfile.write(tar_dir, 48000, y_test[:, T:].reshape(-1))

    plotResult(predictions.reshape(-1), x_test[:, T:], y_test[:, T:].reshape(-1), save_folder)

def checkpoints(model_save_dir, save_folder):
    ckpt_path = os.path.normpath(os.path.join(model_save_dir, save_folder, 'Checkpoints', 'best', 'best.ckpt'))
    ckpt_path_latest = os.path.normpath(
        os.path.join(model_save_dir, save_folder, 'Checkpoints', 'latest', 'latest.ckpt'))
    ckpt_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'Checkpoints', 'best'))
    ckpt_dir_latest = os.path.normpath(os.path.join(model_save_dir, save_folder, 'Checkpoints', 'latest'))

    if not os.path.exists(os.path.dirname(ckpt_dir)):
        os.makedirs(os.path.dirname(ckpt_dir))
    if not os.path.exists(os.path.dirname(ckpt_dir_latest)):
        os.makedirs(os.path.dirname(ckpt_dir_latest))

    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path, monitor='val_loss', mode='min',
                                                       save_best_only=True, save_weights_only=True, verbose=1)
    ckpt_callback_latest = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path_latest, monitor='val_loss',
                                                              mode='min',
                                                              save_best_only=False, save_weights_only=True,
                                                              verbose=1)

    return ckpt_callback, ckpt_callback_latest, ckpt_dir, ckpt_dir_latest