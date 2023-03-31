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


import os
import tensorflow as tf
from LossFunctions import time_loss, freq_loss
from GetData import get_data, get_test_data
from tensorflow.keras.layers import Input, Dense, LSTM, Add, Conv1D
from tensorflow.keras.models import Model
from UtilsForTrainings import plotTraining, writeResults, checkpoints, predictWaves
import pickle

def createModel(T, D, u, k, drop=0.):
    cond_inputs = Input(shape=(D), name='enc_cond')
    encoder_inputs = Input(shape=(T, 1), name='enc_input')
    cond_dense_h = Dense(u, name='Dense_cond_h')(cond_inputs)
    cond_dense_c = Dense(u, name='Dense_cond_c')(cond_inputs)
    state_h = Conv1D(u, k, name='Conv_h')(encoder_inputs)
    state_c = Conv1D(u, k, name='Conv_c')(encoder_inputs)
    states_h = Add()([state_h[:, 0, :], cond_dense_h])
    states_c = Add()([state_c[:, 0, :], cond_dense_c])
    encoder_states = [states_h, states_c]
    decoder_inputs = Input(shape=(T, 1), name='dec_input')
    outputs = LSTM(u, return_sequences=False, return_state=False, name='LSTM_De',
                   dropout=drop)(decoder_inputs, initial_state=encoder_states)
    decoder_outputs = Dense(u, activation='sigmoid', name='DenseLay')(outputs)
    decoder_outputs = Dense(T, name='OutLay')(decoder_outputs)
    model = Model([cond_inputs, encoder_inputs, decoder_inputs], decoder_outputs)
    model.summary()
    return model


def trainED(data_dir, epochs, seed=422, **kwargs):
    ckpt_flag = kwargs.get('ckpt_flag', False)
    b_size = kwargs.get('b_size', 32)
    learning_rate = kwargs.get('learning_rate', 0.001)
    units = kwargs.get('encoder_units', 64)
    model_save_dir = kwargs.get('model_save_dir', './TrainedModels')
    save_folder = kwargs.get('save_folder', 'ED_Testing')
    drop = kwargs.get('drop', 0.)
    loss_type = kwargs.get('loss_type', 'mse')
    w_length = kwargs.get('w_length', 16)
    inference = kwargs.get('inference', False)
    generate_wav = kwargs.get('generate_wav', None)

    T = w_length
    D = 4

    model = createModel(T, D, units, T, drop)
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    if loss_type == 'freq':
        model.compile(loss=freq_loss, metrics=freq_loss, optimizer=opt)
    elif loss_type == 'mse':
        model.compile(loss='mse', metrics=['mse'], optimizer=opt)
    elif loss_type == 'time':
        model.compile(loss=time_loss, metrics=time_loss, optimizer=opt)
    else:
        raise ValueError('Please pass loss_type as either MAE or MSE')

    callbacks = []
    if ckpt_flag:
        ckpt_callback, ckpt_callback_latest, ckpt_dir, ckpt_dir_latest = checkpoints(model_save_dir, save_folder)
        callbacks += [ckpt_callback, ckpt_callback_latest]
        best = tf.train.latest_checkpoint(ckpt_dir_latest)
        if best is not None:
            print("Restored weights from {}".format(ckpt_callback))
            model.load_weights(best)
            # start_epoch = int(latest.split('-')[-1].split('.')[0])
            # print('Starting from epoch: ', start_epoch + 1)
        else:
            print("Initializing random weights.")

    if not inference:
        x, y, cond, x_val, y_val, cond_val = get_data(data_dir=data_dir, window=w_length,
                                                          seed=seed)

        results = model.fit([cond, x[:, :T], x[:, T:]], y[:, T:], batch_size=b_size,
                                    epochs=epochs, verbose=0,
                                    validation_data=([cond_val, x_val[:, :T], x_val[:, T:]], y_val[:, T:]),
                                    callbacks=callbacks)

        writeResults(None, results, b_size, learning_rate, drop, None, loss_type, model_save_dir,
                                   save_folder, 0)

        plotTraining(results.history['loss'], results.history['val_loss'], model_save_dir, save_folder)

    print("Training done")
    x_test, y_test, cond_test = get_test_data(data_dir=data_dir, window=w_length, seed=seed)
    if ckpt_flag:
        best = tf.train.latest_checkpoint(ckpt_dir)
        if best is not None:
            print("Restored weights from {}".format(ckpt_dir))
            model.load_weights(best)

    test_loss = model.evaluate([cond_test, x_test[:, :T], x_test[:, T:]], y_test[:, T:],
                               batch_size=b_size, verbose=0)

    print('Test Loss: ', test_loss)
    if generate_wav is not None:
        predictions = model.predict([cond_test, x_test[:, :T], x_test[:, T:]],
                                    batch_size=b_size)
        predictWaves(predictions, x_test, y_test, model_save_dir, save_folder, T)


    results_ = {'Test_Loss': test_loss}
    print(results_)

    if ckpt_flag:
        with open(os.path.normpath('/'.join([model_save_dir, save_folder, 'results.txt'])), 'w') as f:
            for key, value in results_.items():
                print('\n', key, '  : ', value, file=f)
            pickle.dump(results_, open(os.path.normpath('/'.join([model_save_dir, save_folder, 'results.pkl'])), 'wb'))

    return results_

if __name__ == '__main__':
    data_dir = '../../Files'   # /scratch/users/riccarsi/Files'
    seed = 422
    # start = time.time()
    trainED(data_dir=data_dir,
            model_save_dir='../../TrainedModels',  # '/scratch/users/riccarsi/TrainedModels',
            save_folder='prova',
            ckpt_flag=True,
            b_size=128,
            learning_rate=0.001,
            units=32,
            epochs=10,
            loss_type='freq',
            generate_wav=1,
            w_length=32,
            inference=False)
    # end = time.time()
    # print(end - start)