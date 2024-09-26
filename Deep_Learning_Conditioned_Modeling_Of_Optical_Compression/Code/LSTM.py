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
# R. Simionato, 2023, "Deep Learning Conditioned Modeling of Optical Compression" in proceedings of the 22th Digital Audio Effect Conference, Vienna, Austria.



import numpy as np
import os
import tensorflow as tf
from GetData import get_data
from scipy.io import wavfile
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.models import Model
import pickle

#
def trainLSTM(data_dir, epochs, seed=422, **kwargs):

    
    """
      :param data_dir: the directory in which dataset are stored [string]
      :param batch_size: the size of each batch [int]
      :param learning_rate: the initial leanring rate [float]
      :param units: the number of model's units [array of int]
      :param model_save_dir: the directory in which models are stored [string]
      :param save_folder: the directory in which the model will be saved [string]
      :param inference: if True it skip the training and it compute only the inference [bool]
      :param w_length: input size [int]
      :param epochs: the number of epochs [int]
      :param act: activation function [string]

    """
    
    b_size = kwargs.get('b_size', 128)
    learning_rate = kwargs.get('learning_rate', 0.001)
    units = kwargs.get('units', [32, 32])
    model_save_dir = kwargs.get('model_save_dir', '../../TrainedModels')
    save_folder = kwargs.get('save_folder', 'LSTM')
    opt_type = kwargs.get('opt_type', 'Adam')
    inference = kwargs.get('inference', False)
    loss_type = kwargs.get('loss_type', 'mse')
    w_length = kwargs.get('w_length', 16)
    act = kwargs.get('act', 'tanh')
    scaler = None

    if inference:
        file_scaler = open(os.path.normpath('/'.join([data_dir, 'scaler.pickle'])), 'rb')
        scaler = pickle.load(file_scaler)
        
    # load the data
    x, y, x_val, y_val, x_test, y_test, scaler, fs = get_data(data_dir=data_dir, window=w_length, inference=inference, scaler=scaler,
                                                                              seed=seed)

    layers = len(units)
    n_units = ''
    for unit in units:
        n_units += str(unit) + ', '

    # T past values used to predict the next value
    T = x.shape[1]  # time window
    D = x.shape[2]  # conditioning

    inputs = Input(shape=(T, D), name='enc_input')
    first_unit_encoder = units.pop(0)
    if len(units) > 0:
        last_unit_encoder = units.pop()
        outputs = LSTM(first_unit_encoder, return_sequences=True, stateful=True, name='LSTM_En0')(inputs)
        for i, unit in enumerate(units):
            outputs, state_h, state_c = LSTM(unit, return_sequences=True, stateful=True, name='LSTM_En' + str(i + 1))(outputs)
        outputs = LSTM(last_unit_encoder, stateful=True, name='LSTM_EnFin')(outputs)
    else:
        outputs = LSTM(first_unit_encoder, stateful=True, name='LSTM_En')(inputs)

    outputs = Dense(1, activation=act, name='DenseLay')(outputs)
    model = Model(inputs, outputs)
    model.summary()

    if opt_type == 'Adam':
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif opt_type == 'SGD':
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        raise ValueError('Please pass opt_type as either Adam or SGD')

    if loss_type == 'mae':
        model.compile(loss='mae', metrics=['mae'], optimizer=opt)
    elif loss_type == 'mse':
        model.compile(loss='mse', metrics=['mse'], optimizer=opt)
    else:
        raise ValueError('Please pass loss_type as either MAE or MSE')
     # define callbacks: where to store the weights
    callbacks = []
  
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
    callbacks += [ckpt_callback, ckpt_callback_latest]
    # load the weights of the last epoch, if any
    last = tf.train.latest_checkpoint(ckpt_dir_latest)
    if last is not None:
        print("Restored weights from {}".format(ckpt_dir_latest))
        model.load_weights(last)
    else:
        # if no weights are found,the weights are random generated
        print("Initializing random weights.")

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.000001, patience=20,
                                                               restore_best_weights=True,
                                                               verbose=0)
    callbacks += [early_stopping_callback]
    # train
    if not inference:
        results = model.fit(x, y, batch_size=b_size, epochs=epochs, verbose=0,
                            validation_data=(x_val, y_val), shuffle=False,
                            callbacks=callbacks)

     # load the best weights of the model
    best = tf.train.latest_checkpoint(ckpt_dir)
    if best is not None:
        print("Restored weights from {}".format(ckpt_dir))
        model.load_weights(best).expect_partial()
    else:
        # if no weights are found, there is something wrong
        print("Something is wrong.")

     # compute test loss
    test_loss = model.evaluate(x_test, y_test, batch_size=b_size, verbose=0)
    print('Test Loss: ', test_loss)
    if inference:
        results = {}
    else:
        results = {
            'Test_Loss': test_loss,
            'Min_val_loss': np.min(results.history['val_loss']),
            'Min_train_loss': np.min(results.history['loss']),
            'b_size': b_size,
            'learning_rate': learning_rate,
            'opt_type': opt_type,
            'loss_type': loss_type,
            'w_length': w_length,
            # 'Train_loss': results.history['loss'],
            'Val_loss': results.history['val_loss']
        }
        print(results)
    if not inference:
      
        with open(os.path.normpath('/'.join([model_save_dir, save_folder, 'results.txt'])), 'w') as f:
            for key, value in results.items():
                print('\n', key, '  : ', value, file=f)
            pickle.dump(results,
                        open(os.path.normpath('/'.join([model_save_dir, save_folder, 'results.pkl'])), 'wb'))
            
    if inference:

        # predict the test set and save the results

        predictions = model.predict(x_test)

        print('GenerateWavLoss: ', model.evaluate(x_test, y_test, batch_size=b_size, verbose=0))
        predictions = scaler[0].inverse_transform(predictions)
        x_test = scaler[0].inverse_transform(x_test[:, :, 0])
        y_test = scaler[0].inverse_transform(y_test)

        predictions = predictions.reshape(-1)
        x_test = x_test.reshape(-1)
        y_test = y_test.reshape(-1)
        
        # Define directories
        pred_name = 'LSTM_pred.wav'
        inp_name = 'LSTM_inp.wav'
        tar_name = 'LSTM_tar.wav'

        pred_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', pred_name))
        inp_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', inp_name))
        tar_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', tar_name))

        if not os.path.exists(os.path.dirname(pred_dir)):
            os.makedirs(os.path.dirname(pred_dir))

        # Save Wav files
        predictions = predictions.astype('int16')
        x_test = x_test.astype('int16')
        y_test = y_test.astype('int16')
        wavfile.write(pred_dir, int(fs), predictions)
        wavfile.write(inp_dir, int(fs), x_test)
        wavfile.write(tar_dir, int(fs), y_test)

    return results


if __name__ == '__main__':
    data_dir = './Files' # data folder to dataset
    seed = 422 # seed in case reproducibility is desired
    trainLSTM(data_dir=data_dir,
              model_save_dir='../TrainedModels',
              save_folder='LSTM',
              b_size=128,
              units=[32, 32],
              learning_rate=0.001,
              epochs=100,
              loss_type='mse',
              w_length=1,
              act='tanh',
              inference=False)
