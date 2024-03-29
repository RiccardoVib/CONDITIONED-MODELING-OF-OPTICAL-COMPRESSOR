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
from UtilsForTrainings import plotTraining, writeResults, checkpoints, predictWaves
import pickle

from Models import create_model_ED_CNN
from DatasetsClass import DataGeneratorCL1B, DataGeneratorLA2A
from LossFunctions import STFT
from GetData_test import get_test_data
import numpy as np
import random
    
def train(data_dir, epochs, seed=422, **kwargs):
    ckpt_flag = kwargs.get('ckpt_flag', False)
    b_size = kwargs.get('b_size', 32)
    learning_rate = kwargs.get('learning_rate', 0.0001)
    units = kwargs.get('units', 64)
    d = kwargs.get('cond', 4)
    model_save_dir = kwargs.get('model_save_dir', '/scratch/users/riccarsi/TrainedModels')
    save_folder = kwargs.get('save_folder', 'ED_Testing')
    w_length = kwargs.get('w_length', 16)
    inference = kwargs.get('inference', False)
    activation = kwargs.get('activation', 'sigmoid')
    out = kwargs.get('out', 16)


    #####seed
    np.random.seed(422)
    tf.random.set_seed(422)
    random.seed(422)


    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    T = w_length  # time window
    D = d  # features
    o = out

    model = create_model_ED_CNN(D, T, T, o, units, activation)
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1)

    model.compile(loss='mse', metrics=['mse', 'mae'], optimizer=opt)

    callbacks = []
    if ckpt_flag:
        ckpt_callback, ckpt_callback_latest, ckpt_dir, ckpt_dir_latest = checkpoints(model_save_dir, save_folder)
        callbacks += [ckpt_callback, ckpt_callback_latest]
        best = tf.train.latest_checkpoint(ckpt_dir)
        if best is not None:
            print("Restored weights from {}".format(ckpt_dir))
            model.load_weights(best)
            # start_epoch = int(latest.split('-')[-1].split('.')[0])
            # print('Starting from epoch: ', start_epoch + 1)
        else:
            print("Initializing random weights.")

    w = w_length
    if not inference:
        #train_data
        train_gen = DataGeneratorCL1B("TubeTech_train.pickle", data_dir, input_enc_size=w_length, input_dec_size=w_length, output_size=o, cond_size=D, window=w_length, batch_size=b_size)
        #val_data
        val_gen = DataGeneratorCL1B("TubeTech_val.pickle", data_dir, input_enc_size=w_length, input_dec_size=w_length, output_size=o, cond_size=D, window=w_length, batch_size=b_size)

        results = model.fit(train_gen, batch_size=b_size, epochs=epochs, verbose=0, validation_data=val_gen, callbacks=callbacks)

        writeResults(None, results, b_size, learning_rate, model_save_dir, save_folder, 0)

        loss_training = results.history['loss']
        loss_val = results.history['val_loss']

        plotTraining(loss_training, loss_val, model_save_dir, save_folder)

        print("Training done")

    X, Y, Z = get_test_data(data_dir=data_dir, w=w_length, output_size=o, seed=seed)
    if ckpt_flag:
        best = tf.train.latest_checkpoint(ckpt_dir)
        if best is not None:
            print("Restored weights from {}".format(ckpt_dir))
            model.load_weights(best)
    test_gen = DataGeneratorCL1B("TubeTech_test.pickle", data_dir, input_enc_size=w_length, input_dec_size=w_length, output_size=o, cond_size=D, window=w_length, batch_size=b_size)

    test_loss = model.evaluate(test_gen, batch_size=b_size, verbose=0, return_dict=True)

    print('Test Loss: ', test_loss)
    predictions = model.predict([Z, X[:, :w], X[:, w:]], batch_size=batch_numbers, verbose=0)
        
    predictWaves(predictions, X[:, -o:], Y, model_save_dir, save_folder, 48000)

    results_ = {'Test_Loss': test_loss}
    print(results_)

    if ckpt_flag:
        with open(os.path.normpath('/'.join([model_save_dir, save_folder, 'results.txt'])), 'w') as f:
            for key, value in results_.items():
                print('\n', key, '  : ', value, file=f)
            pickle.dump(results_, open(os.path.normpath('/'.join([model_save_dir, save_folder, 'results.pkl'])), 'wb'))

    return results_


if __name__ == '__main__':
    data_dir = '../../Files'
    seed = 422

    train(data_dir=data_dir,
            model_save_dir='../../TrainedModels',
            save_folder='ED',
            ckpt_flag=True,
            b_size=1,
            learning_rate=0.0001,
            units=32,
            epochs=10,
            activation='sigmoid',
            w_length=16,
            out=16,
            cond=4,
            inference=False)
